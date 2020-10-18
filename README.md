# VISION-AI (Birender Panwar)
### My Projects and experimental works in field of computer vision using deep learning

## Deployed Model and Web Application
Few of below works are hosted on AWS Cloud and is accessible throung website.

Web Application: https://s3.ap-south-1.amazonaws.com/www.aijourney.com/aiwork/vision-ai/index.html


## 1. MONOCULAR DEPTH ESTIMATION AND SEGMENTATION SIMULTANEOUSLY
In this project custom dataset of around half million images are created and various CNN Network are build that can do monocular depth estimation and foreground-background separation simultaneously.

**For Detailed Work:** [(link)](1-monocular-depth-estimation)


## 2. DRIVER ACTIVITY MONITORING
AI based Driver Activity Monitoring Solution to avoid car accidents due to driver distractions. The objective is to build a prediction model that monitors driver activities under varieties of driving conditions and external environment including driving during day time and night time. A predictive system that can recognize any distraction while driving like expression arising due to attending phone calls, testing, drinking, talking to the passenger, or deviating from the standard driving alignment, and this is going to be a multi-class predictive system. This system is useful in the prediction of such anomalies and retrospection of an unlikely and any unfortunate event of accidents

This work is implementing the ResNet50, VGG16, and custom CNN classification network on State Farm Distracted Detection Kaggle case study competition dataset. The model is finally deployed on on Raspberry PI4(RPI4) hardware. 

**For Detailed Work:** [(link)](2-driver-activity-monitoring)

## 3. OBJECT DETECTION USING YOLO-V3
500 unique dataset is collected for class "Donald Duck" whcih is not avaiable in YoloV3. Dataset is self annotated using VGG Annotation tools and trained using Yolo-v3 network.

[![Watch the video](https://img.youtube.com/vi/zkVfTjr1ml4/hqdefault.jpg)](https://youtu.be/zkVfTjr1ml4)

**For Detailed Work:** [(link)](part1/13-object-detection-yolo/part2)

## 4. IMAGE CLASSIFIER AND DEPLOYING MODEL ON AWS LAMBDA USING SERVERLESS FRAMEWORK
In this work, the pretrained MobileNet_V2 network is deployed on AWS Lambda using serverless computing. Serverless framwork manages all the resources in AWS and user need to just focus on their Application and problem solving. All the AWS resocures such as API end point, Lambda functions, Cloud Formations, application packages on S3 and many mores resources are created automatically. It's very cool as it takes all the burden of server resource management from the user.

This work explain step by step proceudre to setup Oracle VM Virtual Box, setting up miniconda environment, installation of npm and serverless packages and finally presenting use of various commands and implementaion of handler function for AWs Lambda.  

**For Detailed Work:** [(link)](part2/1-image-classifier-serverless-aws-lambda)

## 5. DRONE IMAGE CLASSIFICATION
In this work, 20000+ images of 4 different classes of drone are collected and model is build to classify images using Transfer-Learning for Pre-Trained MobileNet_V2 Network.

**For Detailed Work:** [(link)](part2/2-drone-image-classifier)


## 6. FACE ALIGNMENT AND FACE SWAPPING
**Face Align**
Face Alignment using 5-Points Landmark detector by DLIB. In the 5-point model, the landmark points consist of 2 points at the corners of the eye; for each eye and one point on the nose-tip. To perform Face Alignment, first detect 5 Points landmarks and compute the normalized image by using the similarity transform.

**Face Swapping**
It implement face swap model where two front face images of different person is taken as inpted and it generate the output images where first person image is swapped with second person image. It finds 68 points landmarks for both the faces using DLIB. It finds Convex Hull from second image and calculate mask for seamless cloning. It find Delaunay traingulation for convex hull points for both the images. Finally, it apply affine transformation to Delaunay triangles and perform clone seamlessly

**For Detailed Work:** [(link)](part2/3-face-align-and-face-swap)

## 7. FACE RECOGNITION
This is face Recognition solution to recognize 10 Bollywood Stars. Custom dataset is created for Boolywood Stars and are added to LFW-Funneled dataset. The entire entire dataset is then used to build Face Recognition model. Pre-trained model **InceptionRestnetV1** which is trained on **vggface2 dataset** is used and their already learned weights from million of faces are used to build the model. Large Margin Softmax is used which calculate Angular **cosloss** by making use of 128 size embedding vector. Model is finally deployed on AWS lambda 

**For Detailed Work:** [(link)](part2/4-face-recognition)

## 8. HUMAN POSE ESTIMATION AND QNNX PACKAGE
This is HPE solution to predicts 16 joints of the human body. Pre-Trained Resnet50 model on MPII dataset is used and converted into QNNX format and Quantized and finally deployed on AWS Lambda.

Model is converted into ONNX format. ONNX model is quantized to reduce the model file size from 130MB to around 65MB and ONNX Runtine is used for inferencing the model. As target is to deploy the model into AWS Lambda, so no pytorch packages is used here. PIL and Numpy function is used to resizing the image and normalization.
Package dependency: numpy, PIL, OpenCV, onnxruntine
With above apporach, package size for AWS lambda is around 110MB and its well within the limit required for deployment.

**For Detailed Work:** [(link)](part2/5-human-pose-estimation-onnx)

## 9. GENERATIVE ADVERSARIAL NETWORK
This is implementation of GAN with R1 Regularizer to generate India Car images. As part of this work 700+ Indian car dataset is collected and used to build GAN network.
The solution generate the fake car images and generate image is inperpolated for 10 diferent variants.

**For Detailed Work:** [(link)](part2/6-gan-car-images)

## 10. VARIATIONAL AUTO ENCODERS
This is implementation of VAE using KLD loss function to generate India Car images. VAE network is build on 700+ Indian car dataset collected through google images.

**For Detailed Work:** [(link)](part2/7-variational-auto-encoders)

## 11. NEURAL STYLE TRANSFER
This is implementation of Neural Style Transfer Model that take an image and reproduce it with a new artistic style. The algorithm takes three images, an input image, a content-image, and a style-image, and changes the input to resemble the content of the content-image and the artistic style of the style-image

**For Detailed Work:** [(link)](part2/8-neural-style-transfer)

## 12. IMAGE SUPER RESOLUTION MODEL (SRGAN)
This is implementation of Image Super Resolution using SRGAN network for self created Drone images dataset. Model is buit for different upscaler factor of 2 and 4.

**For Detailed Work:** [(link)](part2/8-super-resolution)








