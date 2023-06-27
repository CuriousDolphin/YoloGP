import cv2
import torch
import supervision as sv
import onnxruntime
import numpy as np
import os
from ultralytics import YOLO

print(torch.cuda.is_available())
print(os.getcwd())
print(os.listdir())
video_path = "./video/marquez_onboard.mp4"
webcam = "/dev/video0"
model_path = "./models/yolov8n-100e-motogp.pt"
# define a video capture object

# print(cv2.getBuildInformation())
classes = ["curb", "curb", "helmet", "wheel", "moto", "moto", "rider", "road"]
selected_classes = [0, 1, 2, 3, 4, 5]


class YOLOv8Onnx:
    def __init__(self, model_path: str):
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = onnxruntime.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def predict(self, im, imgsz: int, box_thres: float, iou_thresh: float):
        im_shape = (imgsz, imgsz)
        im = cv2.resize(im, (im_shape[0], im_shape[1]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        im = im / 255.0
        im = im.transpose(2, 0, 1)
        im = im[np.newaxis, :, :, :].astype(np.float32)

        ort_inputs = {self.session.get_inputs()[0].name: im}
        res = self.session.run(None, ort_inputs)[0]
        return res


model = YOLO(model_path)
# model = YOLOv8Onnx(model_path=model_path)
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
vid = cv2.VideoCapture(video_path)
vid.set(3, 640)  # adjust width
vid.set(4, 480)  # adjust height
count = 0
while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    count += 1
    # Display the resulting frame
    if ret:
        detections = []
        frame = cv2.resize(frame, (960, 640))
        res = model(frame, imgsz=(960, 640), conf=0.25, iou=0.45)[0]
        if len(res) > 0:
            detections = sv.Detections.from_yolov8(res)
            detections = detections[np.isin(detections.class_id, selected_classes)]
            helmets_wheel = detections[np.isin(detections.class_id, [2, 3, 4])]
            curbs = detections[np.isin(detections.class_id, [0, 1, 7])]
            # filter_det=[detections.class_id == 2]
            if len(detections) > 0:
                # if len(curbs) > 0:
                #    frame = mask_annotator.annotate(
                #        scene=frame, detections=curbs, opacity=1
                #    )
                frame = box_annotator.annotate(
                    scene=frame,
                    skip_label=True,
                    detections=helmets_wheel,
                    # labels=[classes[cls_id] for cls_id in detections.class_id],
                )
                
        cv2.imshow("frame", frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
