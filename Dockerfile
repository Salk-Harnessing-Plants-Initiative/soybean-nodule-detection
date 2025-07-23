# 1. Base image with Python
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /workspace

# Install system dependencies for OpenCV and others
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy your files into the container
COPY requirements.txt .
COPY src /workspace/src
COPY model /workspace/model
COPY images /workspace/images
COPY images_labels /workspace/images_labels

# 4. Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt 
# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. make the script executable inside the Docker image
RUN chmod +x "src/pipeline.sh"

# 6. add the entrypoint
ENTRYPOINT ["bash", "/workspace/src/pipeline.sh"]
# CMD [ "bash" ]