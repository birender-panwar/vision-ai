from handler import main_handler
import json
import base64
from PIL import Image 
import numpy as np
import io

RESULT_PATH = '.results'


# This fxn convert image to base64 format which is palyable on HTML (see the presence of prefix for HTML)
def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def test_handler():
   
    resp = main_handler(None, '')
    resp_body = json.loads(resp['body'])
    print(resp['statusCode'])
    print(resp_body)
    assert resp['statusCode'] == 200

if __name__ == '__main__':
    print("Running test..")
    test_handler()