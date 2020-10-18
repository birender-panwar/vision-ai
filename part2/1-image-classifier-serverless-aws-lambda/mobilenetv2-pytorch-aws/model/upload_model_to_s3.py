import torch
import boto3


def upload_model(model_path='', s3_bucket='', key_prefix='', aws_profile='default'):
    s3 = boto3.session.Session(profile_name=aws_profile)
    client = s3.client('s3')
    client.upload_file(model_path, s3_bucket, key_prefix)


if __name__ == "__main__":
    S3_BUCKET = 'biru-eva4p2'
    KEY_PREFIX = 'session1/mobilenet_v2.pt'

    SRC_MODEL_PATH = './mobilenet_v2.pt'
    upload_model(SRC_MODEL_PATH, S3_BUCKET, KEY_PREFIX)

    print(SRC_MODEL_PATH + " is uploaded to s3 bucket => " + S3_BUCKET + "/" + KEY_PREFIX) 
