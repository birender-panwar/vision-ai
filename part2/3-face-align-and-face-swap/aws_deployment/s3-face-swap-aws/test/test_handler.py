#import pytest
from handler import main_handler
#from requests_toolbelt import MultipartEncoder
import json
import base64
from PIL import Image 
import numpy as np
import io

DATA_PATH = './images'
RESULT_PATH = '.results'

# Select files for processing
FILE1_PATH = f'{DATA_PATH}/amit.jpg'
FILE2_PATH = f'{DATA_PATH}/arvind.jpg'

# This fxn convert image to base54 format which is palyable on HTML (see the presence of prefix for HTML)
def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def test_handler(filename1, filename2):

    # read and convert image into base64
    img1 = Image.open(filename1)
    img1 = np.array(img1)
    im1_str = img_to_base64(img1).split(',')[1]

    # read and convert image into base64
    img2 = Image.open(filename2)
    img2 = np.array(img2)
    im2_str = img_to_base64(img2).split(',')[1]

    body = json.dumps({"img1": im1_str, "img2": im2_str})
    resp = main_handler({
        'headers': {'content-type': 'application/json'},
        'body': body
    }, '')
    resp_body = json.loads(resp['body'])
    print(resp['statusCode'])
    print(resp_body)
    assert resp['statusCode'] == 200

if __name__ == '__main__':
    print("Running test..")
    test_handler(FILE1_PATH, FILE2_PATH)