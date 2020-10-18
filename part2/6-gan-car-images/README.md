# GENERATIVE ADVERSARIAL NETWORK

**This is implementation of GAN with R1 Regularizer to generate India Car images. GAN network is build on 700+ Indian car dataset collected through google images.**

## Web Application and AWS Lambda Deployment

The model is deployed on AWS Lambda using serverless computing framework and the web application is hosted on AWS S3 bucket

**AWS Deployment Code:** AWS Lambda function and deployment code [(aws_deployment/s6-gan-aws)](aws_deployment/s6-gan-aws)
 
**Web Application:** https://s3.ap-south-1.amazonaws.com/www.aijourney.com/eva4p2/s6/s6_gan.html


## Web App Demonstration

![demo](doc_images/s6_demo_gan.gif)


## Dataset [(link)](https://drive.google.com/file/d/1RT85hbmnCWRHu4Dl9EsJ38urlD1O0KkZ/view?usp=sharing)

700+ Indian car images are collected from google images. For simplicity, car with front facing and specific angle position are collected.

Dataset Size: 704

![sample](doc_images/dataset_samples.jpg)

## GAN Model Creation
 
**Notebook:** /notebooks/S6_R1GAN_Car.ipynb [(Link)](notebooks/S6_R1GAN_Car.ipynb)

**GAN Network:** /notebooks/models/gan_net.py [(Link)](notebooks/models/gan_net.py)

```python
batch_size=64
epochs=1600
n_noise = 256 # noise vector size for Generator
```

**Epoch Results**

![result](doc_images/epoch_results.jpg)

**Real and Fake Discriminitive losses**
![result](doc_images/losses_plot.jpg)

**Generative and Discriminitive Losses**
![result](doc_images/d_g_losses_plot.jpg)

**Interpolation**
Single image is generated and inperpolated for 10 diferent variants
![result](doc_images/interpolation.jpg)



