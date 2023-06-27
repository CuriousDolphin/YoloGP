import gradio as gr

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import supervision as sv
import cv2
import os
import numpy as np

REPO_ID = "CuriousDolphin/yolov8n_motogp"
FILENAME = "yolov8n-seg-100e-motogp-best.pt"
HF_TOKEN = os.getenv("HF_TOKEN")

model = hf_hub_download(
    repo_id=REPO_ID, repo_type="model", filename=FILENAME, token=HF_TOKEN
)
model = YOLO(model)
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
classes = ["curb", "curb", "helmet", "wheel", "moto", "moto", "rider", "road"]
selected_classes = [0, 1, 2, 3, 4, 5, 6]


def inference(image):
    frame = cv2.resize(image, (960, 640))
    res = model(frame, imgsz=(960, 640), conf=0.25, iou=0.45)[0]
    detections = sv.Detections.from_yolov8(res)
    detections = detections[np.isin(detections.class_id, selected_classes)]
    if len(detections) > 0:
        frame = mask_annotator.annotate(scene=frame, detections=detections, opacity=0.8)
        frame = box_annotator.annotate(
            scene=frame,
            skip_label=False,
            detections=detections,
            labels=[classes[cls_id] for cls_id in detections.class_id],
        )
    return frame


with gr.Blocks() as app:
    gr.Markdown("# 🏍️ Motogp tracker")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="numpy")
            button = gr.Button()
        with gr.Column():
            output_im = gr.Image()
        button.click(fn=inference, inputs=[image], outputs=[output_im])

app.queue(concurrency_count=20).launch(server_name="0.0.0.0")
