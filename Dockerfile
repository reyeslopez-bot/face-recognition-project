# Use the latest recommended Python slim image
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run data processing as part of the build
RUN python src/data_processing.py

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set permissions for results directory
RUN mkdir -p /app/results && chmod -R 777 /app/results

CMD ["./entrypoint.sh"]
