#from handler import main_handler
from handler import main_handler_jsondata
import json
import base64
from PIL import Image 
import numpy as np
import io

DATA_PATH = './test_images'
RESULT_PATH = '.results'

# Select files for processing
FILE_PATH = f'{DATA_PATH}/test1.jpg'

# This fxn convert image to base64 format which is palyable on HTML (see the presence of prefix for HTML)
def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def test_handler_formdata(img):
    resp = main_handler({
        'headers': {'content-type': 'multipart/form-data; boundary=X-INSOMNIA-BOUNDARY'},
        'body': img,
    }, '')
    resp_body = json.loads(resp['body'])
    print(resp['statusCode'])
    print(resp_body)
    assert resp['statusCode'] == 200

def test_handler_json(filename):
    # read and convert image into base64
    img = Image.open(filename)
    img = np.array(img)
    im_str = img_to_base64(img).split(',')[1]

    body = json.dumps({"img": im_str})
    resp = main_handler_jsondata({
        'headers': {'content-type': 'application/json'},
        'body': body
    }, '')
    resp_body = json.loads(resp['body'])
    print(resp['statusCode'])
    print(resp_body)
    assert resp['statusCode'] == 200

if __name__ == '__main__':
    print("Running test..")
    test_handler_json(FILE_PATH)