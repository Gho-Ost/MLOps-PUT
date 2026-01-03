from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_s3_notifications as s3_notify,
    Duration,
    RemovalPolicy,
)
from constructs import Construct
import os

class ClassifierStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. Create S3 Bucket where images will be uploaded
        bucket = s3.Bucket(self, "ClassifierBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            cors=[s3.CorsRule(
                allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT, s3.HttpMethods.POST, s3.HttpMethods.DELETE],
                allowed_origins=["*"],
                allowed_headers=["*"]
            )]
        )

        # 2. Define the Docker Image for the Lambda Function
        # The 'code' asset looks for a directory containing a Dockerfile
        # We assume the directory is 'lambda' at the root level of project4
        docker_func = _lambda.DockerImageFunction(self, "ClassifierFunction",
            code=_lambda.DockerImageCode.from_image_asset("lambda"),
            memory_size=2048, # MobileNetV3 is small, but 2GB is safe for inference overhead
            timeout=Duration.seconds(60),
            environment={
                "BUCKET_NAME": bucket.bucket_name
            }
        )

        # 3. Grant the Lambda function permissions to read/write to the bucket
        bucket.grant_read_write(docker_func)

        # 4. Set up the trigger: Invoke Lambda when a .jpg file is uploaded
        bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3_notify.LambdaDestination(docker_func),
            s3.NotificationKeyFilter(suffix=".jpg")
        )
        
        # Adding support for .jpeg and .png as well for convenience, although not strictly required by exam prompt which just says "uploaded". 
        bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3_notify.LambdaDestination(docker_func),
            s3.NotificationKeyFilter(suffix=".jpeg")
        )
        bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3_notify.LambdaDestination(docker_func),
            s3.NotificationKeyFilter(suffix=".png")
        )
