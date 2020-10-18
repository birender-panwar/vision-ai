try:
    import unzip_requirements
except ImportError:
    pass

import torch
import torchvision
from torchvision.transforms import CenterCrop, ToTensor, Resize
from PIL import Image

import boto3
import os
import io
import base64
from requests_toolbelt.multipart import decoder
import json
import numpy as np

print("Import End...")

# define env variables if there are not existing
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'biru-eva4p2'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'session8/srgan_netG_sf2_cpu.pt'

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

g_model = load_model_from_s3()

def tensor2np(tensor):
    img = np.transpose(tensor.detach().numpy(),[1,2,0])
    #img = img.astype(np.float)
    #img = np.clip(img,0,1)
    return img

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_sample_images(real_image, upscale_factor):
    w, h = real_image.size
    crop_size = calculate_valid_crop_size(min(w, h), upscale_factor)
    lr_scale = Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)
    hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
    hr_image = CenterCrop(crop_size)(real_image)
    lr_image = lr_scale(hr_image)
    hr_restore_img = hr_scale(lr_image)
    return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

'''
Input image is received as formdata.
NOTE: In API gateway , Binay Media type shall be set to multipart/form-data
'''
def main_handler(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print('content_type header: ' + content_type_header)
        #print('Event Body: ' + event["body"])

        body = base64.b64decode(event["body"])
        #print('Image content Loaded')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print(f'MultipartDecoder processed')
        img = Image.open(io.BytesIO(picture.content))
        print(f'Input image data loaded')

        lr, hr_restore, hr = get_sample_images(real_image=img, upscale_factor=2)        
        print(f'lr shape: {lr.shape},  hr_restore shape: {hr_restore.shape}, hr shape: {hr.shape}')
        lrSize = lr.shape[1]
        hrSize = hr.shape[1]

        lr = lr.unsqueeze(0)
        sr = g_model(lr)    # create SR image
        sr, lr = sr.squeeze(0), lr.squeeze(0)

        lr_np = tensor2np(lr)
        hr_restore_np = tensor2np(hr_restore)
        sr_np = tensor2np(sr)

        sr_np_norm = np.uint8(sr_np*255)
        print(f'SR image created')

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'lrSize': lrSize, 'hrSize': hrSize, 'lrImg': img_to_base64(lr_np), 'hrRestoredImg': img_to_base64(hr_restore_np), 'srImg': img_to_base64(sr_np_norm)})
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

'''
Input image is received as base64 encoded format in JSON body
'''
def main_handler_jsondata(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print('content_type header: ' + content_type_header)
        #print('Event Body: ' + event["body"])

        # use this when input image is received as JSON body
        json_body = json.loads(event["body"])
        im = base64.b64decode(json_body["img"])
        print('Image content Loaded')

        img = Image.open(io.BytesIO(im))

        lr, hr_restore, hr = get_sample_images(real_image=img, upscale_factor=2)        
        print(f'lr shape: {lr.shape},  hr_restore shape: {hr_restore.shape}, hr shape: {hr.shape}')
        lrSize = lr.shape[1]
        hrSize = hr.shape[1]

        lr = lr.unsqueeze(0)
        sr = g_model(lr)    # create SR image
        sr, lr = sr.squeeze(0), lr.squeeze(0)

        lr_np = tensor2np(lr)
        hr_restore_np = tensor2np(hr_restore)
        sr_np = tensor2np(sr)

        sr_np_norm = np.uint8(sr_np*255)
        print(f'SR image created')


        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'lrSize': lrSize,'hrSize': hrSize,'lrImg': img_to_base64(lr_np), 'hrRestoredImg': img_to_base64(hr_restore_np), 'srImg': img_to_base64(sr_np_norm)})
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
