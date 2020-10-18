import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

import torch.optim as optim
import torch.nn.functional as F

#custom package
from utils.model_history import ModelHistory
from utils.common_utils import LR_UPDATE_TY

class ModelUtils():
    def __init__(self, model, device, train_loader, test_loader, start_epoch, epochs, 
                 criterion, optimizer, lr_scheduler=None, lr_update_ty=None, 
                 reduceLr_scheduler=None,
                 saved_model_dir=None,
                 tqdm_status=True):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_update_ty = lr_update_ty
        self.reduceLr_scheduler = reduceLr_scheduler
        self.saved_model_dir = saved_model_dir
        self.tqdm_status = tqdm_status
        self.history = ModelHistory(start_epoch, epochs)

    def train(self, cur_epoch):
        total_loss = 0
        correct, processed = 0, 0
        acc = 0
        self.model.train()
        pbar = self.train_loader
        if self.tqdm_status:
          pbar = tqdm(self.train_loader)
        
        for batch_idx, (data, target) in enumerate(pbar):
          data, target = data.to(self.device), target.to(self.device)
          self.optimizer.zero_grad() # init gradiant to zero befor starting to do back-propagation
          output = self.model(data) # predict
          loss = self.criterion(output, target) # calculate loss

          total_loss += loss.item() 

          # back-propagation
          loss.backward() 
          self.optimizer.step()

          # read current Lr and update Lr if required
          curLR = self.optimizer.state_dict()['param_groups'][0]['lr']
          if(self.lr_scheduler != None and self.lr_update_ty == LR_UPDATE_TY.BATCHWISE): # for batchwise lr update
            self.lr_scheduler.step()
            curLR = self.lr_scheduler.get_last_lr()[0]

          pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()
          processed += len(data)
          acc = 100*correct/processed

          if self.tqdm_status:
            pbar.set_description(desc=f'Epoch: {cur_epoch} TRAIN ==> [Batch={batch_idx+1}] train_loss={total_loss/(batch_idx+1):0.6f} train_acc: {acc:0.2f} LR={curLR:0.9f}')
          else:
            logfreq = len(self.train_loader)//5
            if ((batch_idx+1) % logfreq == 0):
              print(f'Epoch: {cur_epoch} TRAIN ==> [Batch={batch_idx+1}/{len(self.train_loader)}] train_loss={total_loss/(batch_idx+1):0.6f} train_acc: {acc:0.2f} LR={curLR:0.9f}')
      
        total_loss /= len(self.train_loader)
        #acc = 100. * correct/len(self.train_loader.dataset)
        return np.round(acc,2), np.round(total_loss,6)

    def test(self, cur_epoch):
        total_loss = 0
        correct, processed = 0, 0
        acc = 0
        self.model.eval()

        with torch.no_grad():
          pbar = tqdm(self.test_loader)
          if self.tqdm_status:
            pbar = tqdm(self.test_loader)

          for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data) # predict
            loss = self.criterion(output, target) # calculate loss
            total_loss += loss.item() # sum the batch losses

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            acc = 100*correct/processed

            if self.tqdm_status:
              pbar.set_description(desc=f'Epoch: {cur_epoch} TEST  ==> [Batch={batch_idx+1}] test_loss={total_loss/(batch_idx+1):0.6f} test_acc: {acc:0.2f}')
            else:
              logfreq = len(self.test_loader)//5
              if ((batch_idx+1) % logfreq == 0):
                print(f'Epoch: {cur_epoch} TEST  ==> [Batch={batch_idx+1}/{len(self.test_loader)}] test_loss={total_loss/(batch_idx+1):0.6f} test_acc: {acc:0.2f}')
          
          total_loss /= len(self.test_loader)
          #acc = 100. * correct/len(self.test_loader.dataset)

        return np.round(acc,2), np.round(total_loss,6)

    def build(self):

        max_acc = 0

        for epoch in range(self.start_epoch, self.epochs+1):

          print('\n')

          cur_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

          train_acc, train_loss = self.train(epoch)

          if(self.lr_scheduler != None and self.lr_update_ty == LR_UPDATE_TY.EPOCHWISE):
            self.lr_scheduler.step()

          test_acc, test_loss = self.test(epoch)

          if(self.reduceLr_scheduler != None):
            self.reduceLr_scheduler.step(test_loss)

          if(test_acc > max_acc):
            self.save_model(epoch, test_loss, test_acc)
            max_acc = test_acc

          self.history.append_epoch_result(train_acc, train_loss, test_acc, test_loss, cur_lr)
        
        return self.history

    def save_model(self, epoch, loss, acc):
        if self.saved_model_dir:
          filename = f'{self.saved_model_dir}/ep{epoch}_acc_{acc}_loss_{loss:0.9f}.pth'
          state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'criterion': self.criterion, }
          torch.save(state, filename)
          # Save entire model itself
          #torch.save(self.model, f'{self.saved_model_dir}/model_ep{epoch}_testloss_{loss:0.9f}.pt')
          torch.save(self.model, f'{self.saved_model_dir}/bestmodel.pt')

'''
common functions
'''

def default_build(model, device, train_loader):
    min_lr = 1e-3   # minimum LR 
    L2_val = 1e-4 # L2 Regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=min_lr, momentum=0.9, nesterov=True, weight_decay=L2_val)  

    for epoch in range(0,5):
      total_loss = 0
      correct, processed = 0, 0
      acc = 0
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # init gradiant to zero befor starting to do back-propagation
        output = model(data) # predict
        loss = criterion(output, target) # calculate loss

        total_loss += loss.item() 

        # back-propagation
        loss.backward() 
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        acc = 100*correct/processed
      
      total_loss /= len(train_loader)
      print(f'Epoch: {epoch}, Loss: {loss:9f} Accuracy: {acc:2f}')

def get_test_accuracy(model, device, testloader):
    correct, processed = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

    acc = 100. * correct / processed
    return acc, processed

def get_class_based_accuracy(model, device, testloader, classes):
    num_classes = len(classes)
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        print(f'Accuracy of {classes[i]} : {100*class_correct[i]/class_total[i]:0.2f}%')
    return

# Few common fucntions
def transfer_optimizer(optzr, device):
    # now individually transfer the optimizer parts...
    for state in optzr.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device)

def load_checkpoint(model, optimizer, device, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        criterion = checkpoint['criterion']

        print("=> loading optimizer states..")
        transfer_optimizer(optimizer, device)
        
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, criterion