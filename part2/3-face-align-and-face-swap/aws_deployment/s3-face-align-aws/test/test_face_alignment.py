from PIL import Image
import dlib
import numpy as np
import faceBlendCommon as fbc

import os
import io
import base64
import json

PREDICTOR_PATH = './predictor/shape_predictor_5_face_landmarks.dat'
DATA_PATH = './images'
RESULTS_PATH = './results'

# Initialize the face detector
faceDetector = dlib.get_frontal_face_detector()

# Initialize the Landmark Predictor (a.k.a. shape predictor). The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

def read_image(filename):
    try:
        image = Image.open(filename)
        return np.array(image)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_face_alignment(im):
    # Detect Landmark
    points = fbc.getLandmarks(faceDetector, landmarkDetector, im)
    points = np.array(points)

    # Convert image to floating point in the range 0 to 1
    im = np.float32(im)/255.0

    # Specify the size of aligned face image. Compute the normalized image by using the similarity transform

    # Dimension of output Image
    h = im.shape[0] # 600
    w = im.shape[1] # 600

    # Normalize the image to output coordinates
    imNorm, points = fbc.normalizeImagesAndLandmarks((h,w), im, points)
    imNorm = np.uint8(imNorm*255)

    # This is aligned image
    return imNorm

def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def main_handler(filename):
    try:
        im = read_image(filename)
        aligned_face = get_face_alignment(im)
        
        return {
            "statusCode": 200,
            "body": json.dumps({'file': filename, 'alignedFaceImg': img_to_base64(aligned_face)})
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

if __name__ == "__main__":
    imageFilename = f'{DATA_PATH}/face.jpg'
    res = main_handler(imageFilename)
    print(res)
