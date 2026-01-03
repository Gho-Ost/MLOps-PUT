# Project 3: BentoML Deep Learning Deployment

This project demonstrates how to serve a PyTorch ResNet model using BentoML, containerize it with Docker, and deploy it to a Cloud VM (AWS EC2, Google Cloud, or Azure).

## Prerequisites

-   Python 3.8+ installed locally
-   Docker installed locally (or on the VM)
-   Cloud account (AWS/GCP/Azure) for VM deployment

## Quick Start (Local)

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Model Weights**
    Run the setup script to download ResNet18 weights to your local cache.
    ```bash
    python model_setup.py
    ```

3.  **Run the Service Locally**
    Start the BentoML development server.
    ```bash
    bentoml serve service.py:MobileNetV3Service --reload
    ```
    The service will be available at `http://localhost:3000`.

4.  **Test the Service**
    In a new terminal:
    ```bash
    python client.py
    ```
    You should see prediction results for the test image.

## Deployment Guide

### 1. Build the Bento
Package the model, service code, and dependencies into a standard distribution format (a "Bento").
```bash
bentoml build
```
This will output a tag, e.g., `mobilenet_v3_small_classifier_service:latest`.

### 2. Containerize (Docker)
Convert the Bento into a Docker image.
```bash
DOCKER_BUILDKIT=0 bentoml containerize mobilenet_v3_small_classifier_service:latest
```
*Note: You can verify it works by running `docker run -p 3000:3000 mobilenet_v3_small_classifier_service:latest` locally.*

### 3. Deploy to Cloud VM (e.g., AWS EC2)

#### Option A: Build on VM (Recommended/Easier)
1.  **Launch VM**: Start an Ubuntu instance on AWS/GCP/Azure. ensure port `3000` (or `80` if you bind there) is open in the security group/firewall.
2.  **Transfer Files**: Copy the `project3` folder to the VM using `scp` or `git`.
    ```bash
    # Example SCP command
    scp -i <your-key.pem> -r ./project3 ubuntu@<vm-public-ip>:~/
    ```
3.  **Setup VM**: SSH into the VM and install Docker & Python.
    ```bash
    ssh -i <your-key.pem> ubuntu@<vm-public-ip>
    sudo apt update && sudo apt install -y docker.io python3-pip
    sudo usermod -aG docker $USER
    # Log out and log back in for docker group changes to take effect
    ```
4.  **Install BentoML & Build**:
    ```bash
    cd project3
    
    # Create and activate a virtual environment (fixes PEP 668 error)
    python3 -m venv venv
    source venv/bin/activate
    
    pip3 install -r requirements.txt
    python3 model_setup.py
    bentoml build
    DOCKER_BUILDKIT=0 bentoml containerize mobilenet_v3_small_classifier_service:latest
    ```
5.  **Run Container**:
    ```bash
    docker run -d -p 3000:3000 mobilenet_v3_small_classifier_service:latest
    ```
 
#### Option B: Push Image to Registry (AWS ECR / Docker Hub)
1.  Tag your local image: `docker tag mobilenet_v3_small_classifier_service:latest <your-registry>/mobilenet_v3_small_classifier_service:v1`
2.  Push: `docker push ...`
3.  Pull on VM: `docker run -d -p 3000:3000 <your-registry>/mobilenet_v3_small_classifier_service:v1`

### 4. Verify External Access
From your **local machine**, run the client script pointing to the VM's public IP.

```bash
python client.py http://<vm-public-ip>:3000/classify
```

## Troubleshooting

### Docker Buildx Error (BuildKit)
If you see an error like `BuildKit is enabled but the buildx component is missing`, it means the Docker Buildx plugin is not installed on your system.

**Option 1: Install Buildx (Recommended on Ubuntu)**
```bash
sudo apt-get update
sudo apt-get install -y docker-buildx-plugin
```

**Option 2: Disable BuildKit**
Run the containerize command with `DOCKER_BUILDKIT=0`:
```bash
DOCKER_BUILDKIT=0 bentoml containerize resnet_classifier_service:latest
```
