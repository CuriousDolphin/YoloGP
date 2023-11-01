import os
import torch


from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import cv2
import supervision as sv
from tqdm import tqdm_notebook as tqdm

torch.cuda.is_available()
HOME = os.getcwd()
print(HOME)


SOURCE_IMAGE_PATH = "./dataset/2019_americasGP/"
ontology = CaptionOntology(
    {
        "white curb": "white_curb",
        "red curb": "red_curb",
        "helmet": "helmet",
        "motorcycle wheel": "wheel",
        "motorbike": "moto",
        "motocycle": "moto",
        "road": "road",
    }
)
base_model = GroundedSAM(ontology=ontology)
classes = ontology.classes()
print(classes)
base_model.ontology = ontology
base_model.box_threshold = 0.4
base_model.text_threshold = 0.5

imgs = sv.list_files_with_extensions(directory=SOURCE_IMAGE_PATH)

print(len(imgs))

for im in tqdm(imgs[300:301]):
    im_path = str(im.absolute())
    print(im_path)
    image = cv2.imread(im_path)

    detections = base_model.predict(input=im_path)
    # print(detections)

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections,
        labels=[classes[cls_id] for cls_id in detections.class_id],
    )
    image = mask_annotator.annotate(
        scene=annotated_frame, detections=detections, opacity=0.8
    )
    h, w, c = image.shape
    # image = image[0:h, 220:w]
    sv.plot_image(image, (16, 16))
