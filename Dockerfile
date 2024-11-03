# Use the latest recommended Python slim image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Create directories for data and results
RUN mkdir -p /app/data /app/results

# Run data processing as part of the build to download data and cache it
RUN python src/data_processing.py

# Set default command to run all scripts via entrypoint
CMD ["./entrypoint.sh"]
