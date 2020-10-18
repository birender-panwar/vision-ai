try:
    import unzip_requirements
except ImportError:
    pass

from PIL import Image
import dlib
import numpy as np
import faceBlendCommon as fbc
import cv2

import boto3
import os
import io
import base64
#from requests_toolbelt.multipart import decoder
import json
print("Import End...")


# define env variables if there are not existing
#S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'biru-eva4p2'
#MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'session3/mobilenet_drone.pt'

PREDICTOR_PATH = 'predictor/shape_predictor_68_face_landmarks.dat'

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Initialize the Landmark Predictor (a.k.a. shape predictor). The landmark detector is implemented in the shape_predictor class
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    except Exception as e:
        print(repr(e))
        raise(e)


def get_face_swap(image_bytes1, image_bytes2):
    img1 = transform_image(image_bytes=image_bytes1)
    img2 = transform_image(image_bytes=image_bytes2)

    img1Warped = np.copy(img2)

    # Detect Landmark
    points1 = fbc.getLandmarks(detector, predictor, img1)
    points2 = fbc.getLandmarks(detector, predictor, img2)

    print('Landmarks detected')

    # Find convex hull
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    # Create convex hull lists
    hull1 = []
    hull2 = []
    for i in range(0, len(hullIndex)):
        hull1.append(points1[hullIndex[i][0]])
        hull2.append(points2[hullIndex[i][0]])


    # Calculate Mask for Seamless cloning
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype) 
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    print(f'Calculated Mask for Seamless cloning')

    # Find Centroid
    m = cv2.moments(mask[:,:,1])
    center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))

    # Find Delaunay traingulation for convex hull points
    sizeImg2 = img2.shape    
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = fbc.calculateDelaunayTriangles(rect, hull2)

    # If no Delaunay Triangles were found, quit
    if len(dt) == 0:
        print("ERROR: No Delaunay Triangles were found")
        #quit()

    print(f'Found Delaunay Triangles')

    tris1 = []
    tris2 = []
    for i in range(0, len(dt)):
        tri1 = []
        tri2 = []
        for j in range(0, 3):
            tri1.append(hull1[dt[i][j]])
            tri2.append(hull2[dt[i][j]])

        tris1.append(tri1)
        tris2.append(tri2)

    # Simple Alpha Blending
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(tris1)):
        fbc.warpTriangle(img1, img1Warped, tris1[i], tris2[i])

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

    print(f'Cloned seamlessly')

    # This is swapped image
    return output

def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def main_handler(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print('content_type header: ' + content_type_header)
        print('Event Body: ' + event["body"])

        json_body = json.loads(event["body"])
        im1 = base64.b64decode(json_body["img1"])
        im2 = base64.b64decode(json_body["img2"])

        print('Body Loaded')

        #picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        #print(f'MultipartDecoder processed')

        swapped_face = get_face_swap(image_bytes1=im1, image_bytes2=im2)
        print(f'Face swap processed')

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'file': 'swapped.jpeg', 'swappedFaceImg': img_to_base64(swapped_face)})
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
