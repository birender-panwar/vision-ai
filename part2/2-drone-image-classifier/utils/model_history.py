import numpy as np
import matplotlib.pyplot as plt

class ModelHistory():
  def __init__(self, start_epoch=1, epochs=10):
    self.train_acc = []
    self.train_losses = []
    self.val_acc = []
    self.val_losses = []
    self.lr_values = []
    self.epochs = epochs
    self.start_epoch = start_epoch

  def result_summary(self):

      # best training accuracy
      trainindex = np.array(self.train_acc).argmax()
      print("Training best result: Accuracy: {:0.2f} at Epoch {}".format(self.train_acc[trainindex], trainindex+1))

      # best test accuracy
      testindex = np.array(self.val_acc).argmax()
      print("Testing  best result: Accuracy: {:0.2f} at Epoch {}".format(self.val_acc[testindex], testindex+1))

      print("Acuracy Gap: {:0.2f}".format(np.abs(self.train_acc[trainindex]-self.val_acc[testindex])))
      return

  def append_epoch_result(self, train_acc, train_loss, val_acc, val_loss, lr):
      self.train_acc.append(train_acc)
      self.train_losses.append(train_loss)
      self.val_acc.append(val_acc)
      self.val_losses.append(val_loss)
      self.lr_values.append(lr)
      return

  def getDataSeries(self, name):
      if(name is None):
        return None

      data = None
      if(name == "lr"):
        data = self.lr_values
      elif(name == "val_acc"):
        data = self.val_acc
      elif(name == "val_loss"):
        data = self.val_losses
      elif(name == "train_acc"):
        data = self.train_acc
      elif(name == "train_loss"):
        data = self.train_losses

      return data

  def plot_data_against_epoch(self, title="", seriesname=None, save_filename=None):
      fig, axs = plt.subplots(1,1,figsize=(20,5))

      data = self.getDataSeries(seriesname)
      if(data is None):
        print("invalid dataseries attributes. valid names are: lr, val_acc, val_loss, train_acc, train_loss")
        return

      #data_range = np.array(data).max() - np.array(data).min()
      x_size = len(data)

      axs.plot(range(self.start_epoch,x_size+self.start_epoch), data)

      axs.set_title(title)
      axs.set_ylabel(seriesname)
      axs.set_xlabel("Epochs")
      axs.set_xticks(np.arange(1,x_size+1),x_size/10)
      #axs.set_yticks(np.arange(1,data_range+1),data_range/10)

      plt.show()

      if(save_filename != None):
        fig.savefig(save_filename)
      
      return

  def plot_data_against_lr(self, title="", seriesname=None, save_filename=None):

      fig = plt.figure(figsize=(20,5))

      ydata = self.getDataSeries(seriesname)
      xdata = self.lr_values

      plt.plot(xdata, ydata)

      plt.title(title)
      plt.xlabel('LR')
      plt.ylabel(seriesname)

      plt.show()

      if(save_filename != None):
        fig.savefig(save_filename)

  def plot_history(self, title="", save_filename=None):
      fig, axs = plt.subplots(1,2,figsize=(20,5))
      # summarize history for accuracy
      x_size = len(self.train_acc)

      legend_list = ['train', 'test']

      axs[0].plot(range(self.start_epoch,x_size+self.start_epoch), self.train_acc)
      axs[0].plot(range(self.start_epoch,x_size+self.start_epoch), self.val_acc)

      titlename = '{} - Accuracy'.format(title)
      axs[0].set_title(titlename)
      axs[0].set_ylabel('Accuracy')
      axs[0].set_xlabel('Epoch')
      axs[0].set_xticks(np.arange(1,x_size+1),x_size/10)
      axs[0].legend(legend_list, loc='best')

      # plot losses
      axs[1].plot(range(self.start_epoch,x_size+self.start_epoch),self.train_losses)
      axs[1].plot(range(self.start_epoch,x_size+self.start_epoch),self.val_losses)

      titlename = '{} - Losses'.format(title)
      axs[1].set_title(titlename)
      axs[1].set_ylabel('Loss')
      axs[1].set_xlabel('Epoch')
      axs[1].set_xticks(np.arange(1,x_size+1),x_size/10)
      axs[1].legend(legend_list, loc='best')
      plt.show()

      if(save_filename != None):
        fig.savefig(save_filename)

      return
  
  def plot_loss(self, titlename, save_filename=None):
      fig, axs = plt.subplots(figsize=(14,8))
      # summarize history for accuracy
      x_size = len(self.train_losses)

      legend_list = ['train', 'test']

      # plot losses
      axs.plot(range(self.start_epoch,x_size+self.start_epoch),self.train_losses)
      axs.plot(range(self.start_epoch,x_size+self.start_epoch),self.val_losses)

      axs.set_title(titlename)
      axs.set_ylabel('Loss')
      axs.set_xlabel('Epoch')
      axs.set_xticks(np.arange(1,x_size+1),x_size/10)
      axs.legend(legend_list, loc='best')
      plt.show()

      if(save_filename != None):
        fig.savefig(save_filename)