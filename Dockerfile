# Use the latest recommended Python slim image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Specify the default command (optional)
CMD ["python", "src/model_training.py"]
