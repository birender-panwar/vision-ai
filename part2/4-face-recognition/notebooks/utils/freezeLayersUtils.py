import torch
from torch import nn

def freeze_all_layers(model):
    for name, layer in model.named_modules():
        for parameter in layer.parameters():
          parameter.requires_grad = False

def unfreeze_layers_by_name(model, unfreeze_layer_name_list):
    for name, layer in model.named_modules():
      unfreeze = any(n in name for n in unfreeze_layer_name_list)
      if(unfreeze):
        for parameter in layer.parameters():
          parameter.requires_grad = True
      else:
        for parameter in layer.parameters():
          parameter.requires_grad = False

def unfreeze_layers_by_ids(model, unfreeze_layer_ids_list):
    # MobileNet_V2 have 19 features block accessible throunf name: features[idx]
    for idx in range(19):
      if idx in unfreeze_layer_ids_list:
        for param in model.features[idx].parameters():
          param.requires_grad = True
      else:
        for param in model.features[idx].parameters():
          param.requires_grad = False

def show_layers(model):
    for name, layer in model.named_modules():
      for parameter in layer.parameters():
        print(f'Layer name: {name}, Requires Grad: {parameter.requires_grad}')