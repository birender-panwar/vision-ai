import albumentations
from albumentations.pytorch import ToTensor
from torchvision import transforms
import PIL
import numpy as np
import cv2

'''
Utility functions for Image augmentaion using albumentation package
'''
class AlbumCompose():
    def __init__(self, transform=None):
        self.transform = transform
        
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

class AlbumImageAugmentation():
    def __init__(self, means, stds, size=224):
        self.means =means
        self.stds = stds
        self.size = size

    def get_train_transform(self):

        # fill values for cutout or cropping portion
        fill_value = [255. * mean for mean in self.means]
        rc_padding = 32
        rc_pval = 0.2
        randomCrop = [albumentations.PadIfNeeded(min_height=self.size+rc_padding, min_width=self.size+rc_padding, 
                                                  border_mode=cv2.BORDER_REPLICATE, value=fill_value, p=1.0),
                        
                      albumentations.OneOf([
                                albumentations.RandomCrop(height=self.size, width=self.size, p=rc_pval),
                                albumentations.CenterCrop(height=self.size, width=self.size, p=1-rc_pval),
                              ], p=1.0)
          ]

        train_tf = albumentations.Compose([
                    albumentations.Resize(self.size,self.size),
                    albumentations.RandomBrightness(limit=0.3, p=0.70),
                    albumentations.RandomContrast(limit=0.3, p=0.70),
                    #albumentations.Rotate(limit=(-10,10), p=0.70),
                    randomCrop[0], randomCrop[1],
                    albumentations.HorizontalFlip(p=0.7),
                    #albumentations.ElasticTransform(sigma=50, alpha=1, alpha_affine=10,p=0.10),
                    albumentations.CoarseDropout(max_holes=1, max_height=64, max_width=64, min_height=16, min_width=16, fill_value=fill_value, p=0.70),
                    albumentations.Normalize(mean=self.means, std=self.stds),
                    ToTensor()
        ])

        train_tf = AlbumCompose(train_tf)
        return train_tf

    def get_test_transform(self):
        # Test Phase transformations
        test_transforms = albumentations.Compose([
                                              albumentations.Resize(self.size,self.size),
                                              albumentations.Normalize(mean=self.means, std=self.stds),
                                              ToTensor()
                                              ])
        
        test_tf = AlbumCompose(test_transforms)
        return test_tf

'''
Utility functions for Image augmentaion using Pytorch Transforms package
'''
class PyTorchImageAugmentation():
      def __init__(self, means, stds, size=224):
          self.means =means
          self.stds = stds
          self.size = size

      # Transform functions
      def get_train_transform(self):
          fill_value = [int(255. * mean) for mean in self.means]

          train_tf = transforms.Compose([
                #transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.size, self.size)),
                #transforms.ColorJitter(brightness=0.30, contrast=0.30),
                #transforms.RandomRotation((-5.0, 5.0), fill=(fill_value[0],fill_value[1],fill_value[2])),
                #transforms.RandomCrop(self.size, padding=32, padding_mode='edge'),
                #transforms.RandomCrop(self.size, padding=32, fill=(fill_value[0],fill_value[1],fill_value[2])),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
                #transforms.RandomErasing(scale=(0.02, 0.20), ratio=(0.8, 1.2)),
                #transforms,RandomResizing(scale=(0.02, 0.20), ratio=(0.8, 1.2))                        
          ])
          return train_tf

      def get_test_transform(self):
          test_tf = transforms.Compose([
                #transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds)                
          ])
          return test_tf