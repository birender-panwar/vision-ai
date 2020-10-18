from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import zipfile
import os

def prepare_dataloader(datadir, train_transforms, test_transforms, dataloader_args, valid_size=0.30):
    
    # create dataset from image folder
    train_data = datasets.ImageFolder(datadir,  transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    print(f'Total data: {len(train_data)}\n')

    # splitting dataset into training and validate part 
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)    # shuffling the ids for randomness

    # prepare train and test ids and respective samplers
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    print(f'Training size: {len(train_idx)}')
    print(f'Testing size: {len(test_idx)}')

    # prepare dataloader using sampler
    trainloader = DataLoader(train_data, sampler=train_sampler, **dataloader_args)
    testloader = DataLoader(test_data, sampler=test_sampler, **dataloader_args)
    return trainloader, testloader

'''
dataset count-summary: get items count under each folder 
'''
def get_dataset_count(roordir, classlist):
    total = 0
    for classname in classlist:
      classdir = os.path.join(roordir, classname)
      classcnt = len(os.listdir(classdir))
      total += classcnt
      print(f'{classname}: {classcnt} images')

    print(f'\nTotal images: {total}')
    return

'''
Extract specific zip file on destination folder
'''
def extract_dataset(fn, dest):
    with open(fn, 'rb') as f:
        zf = zipfile.ZipFile(f)
        zf.extractall(dest)