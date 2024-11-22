# Utiliza una imagen base de Python 3.10 slim (más ligera)
FROM python:3.10-slim-bullseye

# Instala las dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt primero para aprovechar la caché de Docker
COPY requirements.txt /app/

# Instala PyTorch (CPU) y otras dependencias
RUN pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de la aplicación
COPY . /app

# Define el comando para ejecutar la aplicación
CMD ["python", "app.py"]
