import json
import boto3
import os
import uuid
import logging
from urllib.parse import unquote_plus
from model_utils import predict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    try:
        logger.info("Received event: " + json.dumps(event, indent=2))
        
        # Get the bucket and object key from the Event
        # We generally handle the first record
        if 'Records' not in event or not event['Records']:
            logger.warning("No records found in event.")
            return {'statusCode': 400, 'body': 'No records found'}

        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            key = unquote_plus(key)
            
            # Skip if this is a result file (infinite loop prevention, though suffix filter in CDK should catch this)
            if "_result.json" in key:
                logger.info(f"Skipping result file {key}")
                continue

            # Download file to /tmp
            download_path = f"/tmp/{uuid.uuid4()}{os.path.splitext(key)[1]}"
            logger.info(f"Downloading {bucket}/{key} to {download_path}")
            s3_client.download_file(bucket, key, download_path)
            
            # Inference
            logger.info("Running inference...")
            # Path to model is in the task root
            model_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', ''), 'mobilenet_v3_small.pt')
            class_id = predict(download_path, model_path=model_path)
            logger.info(f"Inference result: Class ID {class_id}")
            
            # Save result
            result_key = f"{key}_result.json"
            result_body = json.dumps({
                "source_image": key,
                "class_id": class_id,
                "info": "Values correspond to ImageNet 1k classes"
            })
            
            logger.info(f"Uploading result to {bucket}/{result_key}")
            s3_client.put_object(Bucket=bucket, Key=result_key, Body=result_body, ContentType='application/json')
            
            # Clean up /tmp to avoid storage issues on warm starts (optional but good practice)
            try:
                os.remove(download_path)
            except Exception:
                pass

        return {
            'statusCode': 200,
            'body': json.dumps('Inference processing complete')
        }
        
    except Exception as e:
        logger.error(e)
        raise e
