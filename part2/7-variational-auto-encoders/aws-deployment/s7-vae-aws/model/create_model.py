import torch
#from vae_net import ResidualBlock, Decoder

IN_CUDA_MODEL = 'vae_decoder_cuda.pt'

OUT_MODEL_FILE_NAME = 'vae_decoder_cpu.pt'

'''
1. IN_CUDA_MODEL is the model build on colab on GPU.
2. To deploy the model, we need to convert the model for CPU and then deploy.
3. This function is to convert the model trained on GPU (colab) into CPU
'''

def create_model(out_dir='./'):
    model = torch.load(f'./{IN_CUDA_MODEL}', map_location=torch.device('cpu'))
    model = model.to('cpu')
    model.eval()
    #traced_model = torch.jit.trace(model, torch.randn(1,3,224,224))
    #traced_model.save(out_dir + OUT_MODEL_FILE_NAME)
    torch.save(model, out_dir + OUT_MODEL_FILE_NAME)


if __name__ == "__main__":
    create_model()
    print(OUT_MODEL_FILE_NAME + " model file is created")
