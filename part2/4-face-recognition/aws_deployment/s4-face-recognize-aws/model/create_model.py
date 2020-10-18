import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

IN_CUDA_MODEL = 'inceptionresnetv1_fr_cuda.pt'

OUT_MODEL_FILE_NAME = 'inceptionresnetv1_fr.pt'

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
    def __init__(self, loaded_lodel, num_classes=10):
        super(FRConvNetwork, self).__init__()
        self.convlayers = loaded_lodel
        self.fc_final = nn.Linear(3, num_classes)

    def forward(self, x, embed=False):
        x = self.convlayers(x)
        if embed:
            return x
        x = self.fc_final(x)
        x = nn.Softmax(dim=1)(x)
        return x
        
def create_model(out_dir='./'):
    model = torch.load(f'./{IN_CUDA_MODEL}', map_location=torch.device('cpu'))
    model = model.to('cpu')
    model.eval()
    traced_model = torch.jit.trace(model, torch.randn(1,3,224,224))
    traced_model.save(out_dir + OUT_MODEL_FILE_NAME)


if __name__ == "__main__":
    create_model()
    print(OUT_MODEL_FILE_NAME + " model file is created")
