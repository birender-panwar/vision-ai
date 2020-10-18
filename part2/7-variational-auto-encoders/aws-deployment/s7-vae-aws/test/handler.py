try:
    import unzip_requirements
except ImportError:
    pass

import torch
from PIL import Image

import boto3
import os
import io
import base64
import json
import numpy as np

print("Import End...")

# define env variables if there are not existing
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'biru-eva4p2'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'session7/vae_decoder_cpu.pt'

print('Downloading model: ' + MODEL_PATH)

# load the S3 client when lambda execution context is created
s3 = boto3.client('s3')

def load_model_from_s3():
    try:
        if os.path.isfile(MODEL_PATH) != True:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
            print('Creating bytestream')
            bytestream = io.BytesIO(obj['Body'].read())
            print('Loading model')
            model = torch.load(bytestream)
            print('Model loaded')
            return model
        else:
            print('Model loading failed')
    except Exception as e:
        print(repr(e))
        raise(e)

d_model = load_model_from_s3()

def get_sample_image(decoder, latent_dims, n_samples=25):
    n_rows = int(np.sqrt(n_samples))
    sample = torch.randn(n_samples, latent_dims)
    sample = decoder(sample)#.cpu()
    sample = torch.cat([torch.cat([sample[n_rows*j+i] for i in range(n_rows)], dim=1) for j in range(n_rows)], dim=2)
    sample = np.transpose(sample.detach().numpy(),[1,2,0])
    sample = sample.astype(np.float)
    return np.clip(sample,0,1)

def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def main_handler(event, context):
    try:
        print(f'Processing API event')

        # Generate fake images
        fake_images = get_sample_image(d_model, latent_dims=128, n_samples=25)
        fake_images_norm = np.uint8(fake_images*255)

        print(f'Successfully generate fake car images')
        
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'fakeImg': img_to_base64(fake_images_norm)})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
