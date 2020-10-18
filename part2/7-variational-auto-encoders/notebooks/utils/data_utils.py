from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np
import zipfile
import os
import shutil
from pathlib import Path
from tqdm import tqdm

'''
Use this fxn when there is single data folder and there is need to split the dataset as train and val dataset
'''
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
Use this fxn when data is already splitted as train and val folder
'''
def prepare_dataloader_ext(train_dir, test_dir, train_transforms, test_transforms, dataloader_args):
    
    # create dataset from image folder
    train_data = datasets.ImageFolder(train_dir,  transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # prepare dataloader using sampler
    trainloader = DataLoader(train_data, **dataloader_args)
    testloader = DataLoader(test_data, **dataloader_args)
    return trainloader, testloader


'''
dataset count-summary: get items count for specific classes 
'''
def get_dataset_count(rootdir):
    total = 0
    classlist = os.listdir(rootdir)
    for classname in classlist:
      classdir = os.path.join(rootdir, classname)
      classcnt = len(os.listdir(classdir))
      total += classcnt
    print(f'\nTotal images in dataset: {total}')
    return
'''
dataset count-summary: get items count for specific classes 
'''
def get_dataset_count_for_classes(rootdir, classlist):
    total = 0
    for classname in classlist:
      classdir = os.path.join(rootdir, classname)
      classcnt = len(os.listdir(classdir))
      total += classcnt
      print(f'{classname}: {classcnt} images')

    print(f'\nTotal images for specified classes: {total}')
    return

'''
Extract specific zip file on destination folder
'''
def extract_dataset(fn, dest):
    with open(fn, 'rb') as f:
        zf = zipfile.ZipFile(f)
        zf.extractall(dest)

'''
1. Doing train:test split (70:30)
2. As class distribution is not uniform so doing train:test split for each class individuall to ensure that samples from same class exists in both train and val dataset
3. this function create seperate dataset folder with train and val folder inside and all its classes and samples
'''
def prepare_dataset(src_datadir, dst_datadir, valid_size=0.30):

    if not Path(src_datadir).exists():
      print('Missing source folder')
      return
    
    if Path(dst_datadir).exists():
      print('Dataset already exist..')
      return

    classlist = os.listdir(src_datadir)

    # create train and val directory with all the class folder inside it
    Path(dst_datadir).mkdir(exist_ok=True)
    for dtype in ['train', 'val']:
      Path(f'{dst_datadir}/{dtype}').mkdir(exist_ok=True)
      for classname in classlist:
        classdir = f'{dst_datadir}/{dtype}/{classname}'
        Path(classdir).mkdir(exist_ok=True)

    print('Preparing train/val dataset..')

    for classname in classlist:
      imglist = os.listdir(f'{src_datadir}/{classname}')
      imgcnt = len(imglist)
      indices = list(range(imgcnt))      
      split = int(np.floor(valid_size * imgcnt))
      np.random.shuffle(indices)    # shuffling the ids for randomness

      train_idx, test_idx = indices[split:], indices[:split]
      for idx in train_idx:
        srcfile = f'{src_datadir}/{classname}/{imglist[idx]}'
        dstdir = f'{dst_datadir}/train/{classname}'
        shutil.copy(srcfile, dstdir)

      for idx in test_idx:
        srcfile = f'{src_datadir}/{classname}/{imglist[idx]}'
        dstdir = f'{dst_datadir}/val/{classname}'
        shutil.copy(srcfile, dstdir)
    print('Dataset is created..')

def get_dataset_summary(roordir):
    ds_summary = {"total": 0, "train": 0, "val": 0, "num_classes":0}
    for dtype in ['train', 'val']:
      classlist = os.listdir(f'{roordir}/{dtype}')
      for classname in classlist:
        classdir = f'{roordir}/{dtype}/{classname}'
        cnt = len(os.listdir(classdir))
        ds_summary[dtype] += cnt

    ds_summary["total"] = ds_summary["train"] + ds_summary["val"]
    ds_summary["num_classes"] = len(os.listdir(f'{roordir}/train'))
    return ds_summary

'''
it scan all files within the class folder and check if all the images can be read as image file..
return all the error files as list
'''
def scan_files_validity(rootdir, classname):
    invalid_files = []
    classdir = os.path.join(rootdir, classname)
    pbar = tqdm(Path(f'{classdir}/').glob('*.*'))
    for p in pbar:
      try:
        #print(f'image : {p}')
        im = Image.open(p)
        im2 = im.convert('RGB')
        #print(np.array(im2).shape)
      except OSError:
        print(f'Cannot load : {p}')
        invalid_files.append(p)

      pbar.set_description(desc=f'Scanning {classname} images..')
    return invalid_files

'''
it scan all files and check if all the images can be read as image file..
return all the error files as list
'''
def scan_dataset(rootdir, classlist):
    errfiles = []
    for classname in classlist:
      invalid_files = scan_files_validity(rootdir, classname)
      errfiles.extend(invalid_files)
    return errfiles