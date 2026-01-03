# Project 4: Serverless Image Classification Instructions

## Overview
This project deploys a serverless image classification system using AWS CDK (Python), Lambda, and S3.
It uses MobileNetV3 Small for inference suitable for limited resource environments.

## Prerequisites
- AWS CLI configured (`aws configure`)
- AWS CDK CLI installed (`npm install -g aws-cdk`)
- Docker installed and running
- Python 3.9+ and pip installed

## Deployment Steps

1. **Initialize Environment**
   Navigate to the project directory:
   ```bash
   cd project4
   ```

2. **Install Python Dependencies**
   Install the CDK app dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   (Optional) You may want to create a virtual environment first:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Bootstrap CDK (First Time Only)**
   If you have never used CDK in this AWS region before:
   ```bash
   cdk bootstrap
   ```

4. **Deploy the Stack**
   This command will build the Docker image and deploy the infrastructure:
   ```bash
   cdk deploy
   ```
   Confirm the changes when prompted.

   > **Note**: The Docker build process downloads the model weights. This might take a minute.

## Usage

1. **Upload an Image**
   Once deployed, find the Bucket Name in the CDK outputs (or check AWS Console).
   Upload a `.jpg` image to the bucket:
   ```bash
   aws s3 cp my_cat.jpg s3://<BucketName>/
   ```

2. **Check the Result**
   Wait a few seconds for the Lambda to process the image. A new file ending in `_result.json` will appear:
   ```bash
   aws s3 ls s3://<BucketName>/
   aws s3 cp s3://<BucketName>/my_cat.jpg_result.json .
   cat my_cat.jpg_result.json
   ```

3. **Interpret the Result**
   The result contains a numeric `class_id`. To get the human-readable class name, use the provided helper script:
   ```bash
   python get_class_name.py my_cat.jpg_result.json
   ```
   Example output:
   ```
   Class ID: 285
   Class Name: Egyptian Cat
   ```

## Testing Locally (Unit Tests)

To run the provided unit tests:
```bash
# Install test dependencies
pip install pytest boto3 torch torchvision pillow

# Run tests
pytest tests/unit/
```

## Cleanup

To remove all resources and avoid costs:
```bash
cdk destroy
```
Note: The S3 bucket is configured to auto-delete objects on destroy, so all data will be lost.
