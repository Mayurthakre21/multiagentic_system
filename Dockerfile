# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Run Flask app using waitress
CMD ["python", "app.py"]
