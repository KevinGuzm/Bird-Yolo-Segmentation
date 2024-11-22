# Usa una imagen base de Python 3.10
FROM python:3.10

# Instala las dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos locales al directorio de trabajo del contenedor
COPY . /app

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Define el comando para ejecutar la aplicación
CMD ["python", "app.py"]
