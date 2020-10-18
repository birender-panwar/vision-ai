import torch
from torchvision.models import resnet

MODEL_FILE_NAME = 'resnet34.pt'

def create_model(out_dir='./'):
    model = resnet.resnet34(pretrained=True)
    model.eval()
    traced_model = torch.jit.trace(model, torch.randn(1,3,224,224))
    traced_model.save(out_dir + MODEL_FILE_NAME)


if __name__ == "__main__":
    create_model()
    print(MODEL_FILE_NAME + " model file is created")
