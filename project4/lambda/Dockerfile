FROM public.ecr.aws/lambda/python:3.11

# Update pip to avoid metadata generation errors
RUN pip install --upgrade pip

# Install PyTorch specifically for CPU to save space and avoid conflicts
# We use the official PyTorch wheel index for CPU
# Install PyTorch specifically for CPU to save space and avoid conflicts
# We use the official PyTorch wheel index for CPU
# We install numpy first from PyPI ensuring we get a binary wheel (avoiding build issues)
# Pinning numpy<2 for stability and forcing binary
RUN pip install "numpy<2" --target "${LAMBDA_TASK_ROOT}" --only-binary=:all:
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --target "${LAMBDA_TASK_ROOT}" --only-binary=:all:

# Install other dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy model download script and run it to bake weights into image
COPY download_model.py .
# Ensure we run this with the installed dependencies in the path
RUN PYTHONPATH="${LAMBDA_TASK_ROOT}" python download_model.py

# Copy function code
COPY handler.py model_utils.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "handler.lambda_handler" ]
