try:
    import unzip_requirements
except ImportError:
    pass

import torch
import torchvision
import torchvision.transforms as transforms
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
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'session4/inceptionresnetv1_fr.pt'

class_names = ['Akshay Kumar', 'Amitabh Bachchan', 'Amrish Puri', 'Anil Kapoor', 'Kajol', 'Katrina Kaif', 'Madhuri Dixit', 'Rajesh Khanna', 'Shilpa Shetty', 'Vinod Khanna']

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
            model = torch.jit.load(bytestream)
            print('Model loaded')
            return model
        else:
            print('Model loading failed')
    except Exception as e:
        print(repr(e))
        raise(e)

model = load_model_from_s3()


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()

def get_top_predictions(model, image_bytes, class_namelist):
    tensor = transform_image(image_bytes=image_bytes)
    scores = model(tensor)[0].tolist() # predict class

    # sorting scores and get top 5 predictions
    npscores = np.array(scores)
    sorted_pred_idx = npscores.argsort()[::-1][:5] # top 5
    pred_classes = []

    for i in range(len(sorted_pred_idx)):
      idx = sorted_pred_idx[i]
      pred_classes.append({"name": class_namelist[idx], "score": np.round(scores[idx], 4)})
    return pred_classes

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
        print('Image content Loaded')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print(f'MultipartDecoder processed')
        predictions = get_top_predictions(model=model, image_bytes=picture.content, class_namelist=class_names)        
        print(f'Predictions: {predictions}')

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predictions': predictions})
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
        print('Event Body: ' + event["body"])

        # use this when input image is received as JSON body
        json_body = json.loads(event["body"])
        im = base64.b64decode(json_body["img"])
        print('Image content Loaded')
        predictions = get_top_predictions(model=model, image_bytes=im, class_namelist=class_names)
        print(f'Predictions: {predictions}')

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'predictions': predictions})
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