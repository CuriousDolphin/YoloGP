import numpy as np

def xyxy2xywh(x) -> np.ndarray:
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    xyxy = np.copy(x)
    xyxy[0] = (x[0] + x[2]) / 2  # x center
    xyxy[1] = (x[1] + x[3]) / 2  # y center
    xyxy[2] = x[2] - x[0]  # width
    xyxy[3] = x[3] - x[1]  # height
    return xyxy


def xywh2xyxy(x) -> np.ndarray:
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def nms(boxes, scores, iou_threshold: float) -> list[int]:
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes) -> float:
    # Compute Intersection over Union (IoU) between a single box and a set of boxes.
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def rescale_boxes(boxes, input_width, input_height, im_width, im_height) -> np.ndarray:
    # Rescale boxes to original image dimensions
    input_shape = np.array(
        [
            input_width,
            input_height,
            input_width,
            input_height,
        ]
    )
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([im_width, im_height, im_width, im_height])
    return boxes