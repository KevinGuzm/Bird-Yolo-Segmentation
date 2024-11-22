import cv2
from ultralytics import YOLO
import gradio as gr
import tempfile
import os

# Cargar el modelo YOLOv8
model = YOLO("./lastmodel.pt")

def process_video(video_input):
    # Verificar el tipo de entrada y obtener la ruta del video
    if isinstance(video_input, dict):
        video_path = video_input.get('name', None)
    elif isinstance(video_input, str):
        video_path = video_input
    else:
        video_path = None

    if not video_path or not os.path.exists(video_path):
        print(f"Error: El archivo {video_path} no existe.")
        return None

    # Crear un archivo temporal para el video de salida
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # Abrir el archivo de video
    cap = cv2.VideoCapture(video_path)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return None

    # Obtener las propiedades del video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el video de salida
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # Crear un VideoWriter para guardar el video procesado
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Iterar a través de los fotogramas del video
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Ejecutar la inferencia de YOLOv8 en el fotograma
        results = model(frame)

        # Visualizar los resultados en el fotograma
        annotated_frame = results[0].plot()

        # Escribir el fotograma anotado en el archivo de salida
        out.write(annotated_frame)

    # Liberar recursos
    cap.release()
    out.release()

    return output_path

# Ruta al video de ejemplo
example_video = "./855082-hd_1920_1080_25fps.mp4"

# Crear la interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Detección de Segmentos de Aves")
    gr.Markdown("Sube un video y detecta segmentos de aves usando YOLOv8, o selecciona el video de ejemplo.")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Video de Entrada")
            process_button = gr.Button("Procesar Video")

            # Añadir el video de ejemplo
            gr.Examples(
                examples=[[example_video]],
                inputs=video_input,
                outputs=None,
                label="Video de Ejemplo"
            )
        with gr.Column():
            video_output = gr.Video(label="Video con Detecciones")

    process_button.click(fn=process_video, inputs=video_input, outputs=video_output)

demo.launch(share=True)
