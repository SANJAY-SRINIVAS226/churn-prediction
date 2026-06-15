# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install system tools and Python dependencies cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# 🌟 FIXED: Copy your optimized preprocessing and model assets directly 
# from your local project's 'artifacts' folder into the container's flat /app/model folder.
# This prevents hardcoded MLflow run-hash string failures completely!
COPY artifacts/ /app/model/
# Safely duplicate the artifact model folder structure to match your fallback paths
COPY artifacts/ /app/src/serving/model/

# Configure environmental variables
# PYTHONPATH=/app ensures you can import 'src' modules smoothly inside the Linux container
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app

# 6. Expose FastAPI port
EXPOSE 8000

# 7. Run the FastAPI app using uvicorn
# 🌟 FIXED: Make sure "app.py" or "main.py" matches your actual FastAPI filename!
# If your serving file is called app.py inside src/, change "main:app" to "app:app"
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
