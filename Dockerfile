FROM python:3.10-slim

WORKDIR /app

# OpenCV / video runtime deps (libgl1-mesa-glx is gone on Debian bookworm+)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5900

ENV FLASK_APP=app.py
ENV FLASK_DEBUG=0

CMD ["python", "app.py"]
