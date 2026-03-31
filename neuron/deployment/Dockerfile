# 1. Start with a lightweight, official Python base image
FROM python:3.10-slim

# 2. Set the directory inside the container where our app will live
WORKDIR /app

# 3. Install the CPU-only version of PyTorch first to save massive amounts of space
# (The GPU version is over 2GB; the CPU version is ~150MB)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Copy the requirements file and install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your project files into the container
# This includes main.py, index.html, and all your .bin, .json, and .pth model files
COPY . .

# 6. Expose the port FastAPI uses
EXPOSE 8000

# 7. The command to start the server when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]