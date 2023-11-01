import gradio as gr

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import supervision as sv
import cv2
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
REPO_ID = "CuriousDolphin/yolov8n_motogp"
FILENAME = "yolov8n-seg-100e-motogp-best.pt"
HF_TOKEN = os.getenv("HF_TOKEN")


model_path = hf_hub_download(
    repo_id=REPO_ID, repo_type="model", filename=FILENAME, token=HF_TOKEN
)
model = YOLO(model_path)
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator(opacity=0.8)
classes = ["curb", "curb", "helmet", "wheel", "moto", "moto", "rider", "road"]
selected_classes = [0, 2, 3, 5, 6]


def inference(image, conf: float, iou: float, progress=gr.Progress()):
    frame = cv2.resize(image, (960, 640))
    res = model(frame, imgsz=(960, 640), conf=conf, iou=iou)[0]
    detections = sv.Detections.from_ultralytics(res)
    detections = detections[np.isin(detections.class_id, selected_classes)]
    if len(detections) > 0:
        frame = mask_annotator.annotate(scene=frame, detections=detections)
        frame = box_annotator.annotate(
            scene=frame,
            skip_label=False,
            detections=detections,
            labels=[classes[cls_id] for cls_id in detections.class_id],
        )
    return frame


with gr.Blocks() as inference_app:
    gr.Markdown("# 🏍️ YoloGP: Motogp tracker")
    with gr.Row():
        with gr.Column():
            image = gr.Image()
            conf = gr.Slider(label="Confidence", minimum=0, maximum=0.99, value=0.3)
            iou = gr.Slider(label="IoU", minimum=0, maximum=0.99, value=0.45)

            with gr.Row():
                button = gr.Button(variant="primary")
            examples = gr.Examples(
                examples=[
                    ["./assets/Rossi_Lorenzo_Catalunya2009.png"],
                    ["./assets/sample1.png"],
                ],
                inputs=[image],
            )
        with gr.Column():
            output_im = gr.Image()

    button.click(fn=inference, inputs=[image, conf, iou], outputs=output_im)

if __name__ == "__main__":
    inference_app.queue(max_size=10).launch(server_name="0.0.0.0")
