import dlib
import numpy as np
import utils.faceBlendCommon as fbc
from PIL import Image

class AlignFace(object):
    def __init__(self, detector, predictor):
        self.detector = detector
        self.predictor = predictor

    def __call__(self, img):
        try: 
          im = self.get_face_alignment(img)
          return im
        except Exception as e:
          return img

    def get_face_alignment(self, im):
        im = np.array(im)

        # Detect Landmark
        points = fbc.getLandmarks(self.detector, self.predictor, im)
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
        return Image.fromarray(imNorm)

def init_face_align(pridictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pridictor_path)
    return detector, predictor