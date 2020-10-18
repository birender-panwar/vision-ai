import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import utils.common_utils as cutils

class PlotUtils(object):
	def __init__(self, device, classes, means=None, stds=None):
	  self.device = device
	  self.classes = classes
	  self.means = means
	  self.stds = stds
		  
	# functions to show an image after de-normalizing
	def imshow_ext(self, img):
		#img = img / 2 + 0.5   # first convert back to [0,1] range from [-1,1] range
		unnormalized = cutils.UnNormalize(mean=self.means, std=self.stds)
		img = unnormalized(img)
		npimg = img.numpy()
		# convert from CHW to HWC
		plt.imshow(np.transpose(npimg, (1, 2, 0)))

	# functions to show an image
	def imshow(self, img):
		#img = img / 2 + 0.5   # first convert back to [0,1] range from [-1,1] range
		npimg = img.numpy()
		# convert from CHW to HWC
		# from 3x32x32 to 32x32x3
		plt.imshow(np.transpose(npimg, (1, 2, 0)))

	def plot_dataset_images(self, data_loader, figsize=(10,9), num_of_images=20, save_filename=None):
		cnt = 0
		fig = plt.figure(figsize=figsize)
		for data, target in data_loader:
			data, target = data.to(self.device), target.to(self.device)
			for index, label in enumerate(target):
				title = "{}".format(self.classes[label.item()])
				ax = fig.add_subplot(4, 5, cnt+1, xticks=[], yticks=[])
				ax.axis('on')
				ax.set_title(title)
				if(self.means is not None and self.stds is not None):
					self.imshow_ext(data[index].cpu())
				else:
					self.imshow(data[index].cpu())
			  
				cnt += 1
				if(cnt==num_of_images):
					break
			if(cnt==num_of_images):
				break
		
		if save_filename:
			fig.savefig(save_filename)
		return

	def plot_dataset_images_ext(self, data_loader, figsize=(10,9), num_of_images=20, save_filename=None):
		cnt = 0
		fig = plt.figure(figsize=figsize)
		for data in data_loader:
			data = data.to(self.device)
			for index, img in enumerate(data):
				ax = fig.add_subplot(4, 5, cnt+1, xticks=[], yticks=[])
				ax.axis('on')
				if(self.means is not None and self.stds is not None):
					self.imshow_ext(img.cpu())
				else:
					self.imshow(img.cpu())
			  
				cnt += 1
				if(cnt==num_of_images):
					break
			if(cnt==num_of_images):
				break
		
		if save_filename:
			fig.savefig(save_filename)
		return
		
	def plot_misclassified_images(self, model, testloader, figsize=(15,12), num_of_images = 20, nrow=4, save_filename=None):
		model.eval()
		misclassified_cnt = 0
		ncol = num_of_images/nrow
		fig = plt.figure(figsize=figsize)
		for data, target in testloader:
			data, target = data.to(self.device), target.to(self.device)
			output = model(data)
			pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
			pred_marker = pred.eq(target.view_as(pred))   
			wrong_idx = (pred_marker == False).nonzero()  # get indices for wrong predictions
			for idx in wrong_idx:
				index = idx[0].item()
				title = "T:{}, P:{}".format(self.classes[target[index].item()], self.classes[pred[index][0].item()])
				ax = fig.add_subplot(nrow, ncol, misclassified_cnt+1, xticks=[], yticks=[])
				#ax.axis('off')
				ax.set_title(title)
				if(self.means is not None and self.stds is not None):
					self.imshow_ext(data[index].cpu())
				else:
					self.imshow(data[index].cpu())
				misclassified_cnt += 1
				if(misclassified_cnt==num_of_images):
					break
			if(misclassified_cnt==num_of_images):
			  break

		if save_filename:
			fig.savefig(save_filename)
		return

	def plot_misclassified_images_for_class(self, model, testloader, classid, figsize=(15,12), num_of_images = 20, nrow=4, save_filename=None):
		model.eval()
		misclassified_cnt = 0
		ncol = num_of_images/nrow
		fig = plt.figure(figsize=figsize)
		for data, target in testloader:
			data, target = data.to(self.device), target.to(self.device)
			output = model(data)
			pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
			pred_marker = pred.eq(target.view_as(pred))   
			wrong_idx = (pred_marker == False).nonzero()  # get indices for wrong predictions
			for idx in wrong_idx:
				index = idx[0].item()
				if(target[index].item() == classid):
					title = "T:{}, P:{}".format(self.classes[target[index].item()], self.classes[pred[index][0].item()])
					ax = fig.add_subplot(nrow, ncol, misclassified_cnt+1, xticks=[], yticks=[])
					#ax.axis('off')
					ax.set_title(title)
					if(self.means is not None and self.stds is not None):
						self.imshow_ext(data[index].cpu())
					else:
						self.imshow(data[index].cpu())
					misclassified_cnt += 1
					if(misclassified_cnt==num_of_images):
						break
			if(misclassified_cnt==num_of_images):
			  break

		if save_filename:
			fig.savefig(save_filename)
		return

	'''
	Added for plotting images using grid package
	'''
	def visualize_data(self, im_data, figsize=(24,8), filename=None, show=True, nrow=4):
		try:
			im_data = im_data[:16]
			im_data = im_data.detach().cpu()
		except:
			pass

		grid_tensor = torchvision.utils.make_grid(im_data, nrow=nrow)
		grid_imgs = grid_tensor.permute(1,2,0)
		plt.figure(figsize=figsize)
		plt.imshow(grid_imgs)
		plt.axis("off")
		if filename:
			plt.savefig(filename)   
		if show:   
			plt.show()
		plt.clf()

	def visualize_data_norm(self, im_data, means, stds, figsize=(24,8), filename=None, show=True, nrow=4):
		im_data = im_data.detach().cpu()
		im_data = (im_data*stds[None, :, None, None]) + means[None, :, None, None]
		self.visualize_data(im_data, figsize=figsize, filename=filename, show=show, nrow=nrow)
