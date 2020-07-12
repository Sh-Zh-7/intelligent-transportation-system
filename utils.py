"""
"""

import argparse
import numpy as np
from PIL import Image

from Detection.traffic_light import get_traffic_light_color
from Tracking.DeepSort.deep_sort import preprocessing
from Tracking.DeepSort.deep_sort.detection import Detection

from visualize import color_str_tuple_map

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, default="./Videos/video-01.avi", help="position of your video")
    parser.add_argument("--input_background", type=str, default="./Videos/background_1.png", help="position")
    parser.add_argument("--output_dir", type=str, default="./Result",
                        help="position to store each frame and result video")

    args = parser.parse_args()
    return args


def update_tracker(image, detection_model, encoder, tracker_model, nms_max_overlap):
    # Start object detection
    img = Image.fromarray(image[..., ::-1])
    boxs, class_names, confidence = detection_model.detect_image(img)
    features = encoder(image, boxs)
    detections = [Detection(bbox, confidence, feature, class_name) for bbox, feature, class_name in
                  zip(boxs, features, class_names)]
    # Run NMS
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    # Call the tracker
    tracker_model.predict()
    tracker_model.update(detections)


def get_environment(image, traffic_lights_bboxes):
    for bbox in traffic_lights_bboxes:
        image = np.array(image)
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        roi = image[y:y + h, x:x + w, :]
        color_str = get_traffic_light_color(roi)
        color = color_str_tuple_map[color_str]
        return color
