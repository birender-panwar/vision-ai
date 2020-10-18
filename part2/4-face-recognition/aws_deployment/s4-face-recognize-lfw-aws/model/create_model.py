import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.lsoftmax import LSoftmaxLinear

IN_CUDA_MODEL = 'inceptionresnetv1_fr_lfw_cuda.pt'

OUT_MODEL_FILE_NAME = 'inceptionresnetv1_fr_lfw.pt'

'''
1. mobilenet_drone_cuda.pt is the model uild on colab on GPU.
2. To deploy the model, we need to convert the model for CPU and then deploy.
3. This function is to convert the model trained on GPU (colab) into CPU
'''

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

class FRConvNetwork(nn.Module):
    def __init__(self, loaded_lodel, margin, device, embedding_size, num_classes=10):
        super(FRConvNetwork, self).__init__()
        self.loaded_lodel = loaded_lodel
        self.margin = margin
        self.device = device

        self.lsoftmax_linear = LSoftmaxLinear(
            input_features=embedding_size, output_features=num_classes, margin=margin, device=device)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()

    def forward(self, x, target=None, embed=False):
        x = self.loaded_lodel(x)     
        if embed:
            return x   
        logit = self.lsoftmax_linear(input=x, target=target)
        return logit
        
def create_model(out_dir='./'):
    model = torch.load(f'./{IN_CUDA_MODEL}', map_location=torch.device('cpu'))
    model = model.to('cpu')
    model.eval()
    traced_model = torch.jit.trace(model, torch.randn(1,3,224,224))
    traced_model.save(out_dir + OUT_MODEL_FILE_NAME)


if __name__ == "__main__":
    create_model()
    print(OUT_MODEL_FILE_NAME + " model file is created")
