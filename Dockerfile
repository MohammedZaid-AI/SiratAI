# Use official Python image
FROM python:3.10-slim

# Install OS dependencies needed for torch + sentence-transformers
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy all project files
COPY . .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# HF Spaces runs apps on port 7860
ENV PORT 7860

# Expose port
EXPOSE 7860

# Launch Flask app using Gunicorn (production server)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
