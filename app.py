import gradio as gr
import cv2
from ultralytics import YOLO
import os

# Ruta al modelo y al video de ejemplo
model_path = "./lastmodel.pt"
example_video_path = "./855082-hd_1920_1080_25fps.mp4"

# Cargar el modelo YOLOv8
model = YOLO(model_path)

# Función para procesar el video y devolver el anotado
def process_video(video_file):
    # Leer el video cargado
    cap = cv2.VideoCapture(video_file.name)
    
    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Crear un archivo de salida para el video anotado
    output_path = "output_annotated_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Ejecutar inferencia en cada fotograma
        results = model(frame)

        # Visualizar los resultados en el fotograma
        annotated_frame = results[0].plot()

        # Guardar el fotograma anotado en el archivo de salida
        out.write(annotated_frame)

    # Liberar recursos
    cap.release()
    out.release()

    return output_path  # Devolver la ruta al video anotado

# Configurar ejemplos
example_videos = [[example_video_path]]

# Crear la interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("# YOLOv8 Video Segmentation")
    gr.Markdown("Upload a video or select the example video below to run YOLOv8 segmentation.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video", type="file")
            process_button = gr.Button("Run YOLOv8")
        with gr.Column():
            video_output = gr.Video(label="Annotated Video")
    
    gr.Examples(
        examples=example_videos,
        inputs=[video_input],
        label="Example Video"
    )

    process_button.click(fn=process_video, inputs=[video_input], outputs=[video_output])

# Ejecutar la aplicación
if __name__ == "__main__":
    demo.launch(share=True)
