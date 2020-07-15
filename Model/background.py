import cv2
from PIL import Image

from Model.Segmentation.zebra_crossing import get_zebra_crossing, get_zebra_area
from Model.Tradition.lane_detect import lane_detect, get_standard_lane_marks

standard_lane_marks = get_standard_lane_marks("./Model/Tradition/standard_lane_marks/")


def get_zebra_rect(img, model):
    """
    Get zebra rect(opencv RotatedRect type)
    :param model: segmentation network
    :param img: origin image
    :return: zebra rect
    """
    result = get_zebra_crossing(img, model)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    zebra_box, zebra_rect = get_zebra_area(gray)
    return zebra_rect


def get_traffic_light(img, model):
    """
    Using model to get traffic light
    :param img: origin image
    :param model: detection model
    :return: traffic light bboxes
    """
    img = Image.fromarray(img[..., ::-1])
    model.set_detection_class(["traffic light"])
    traffic_light_boxes, _, __ = model.detect_image(img)
    return traffic_light_boxes


def static_process(image_path, object_detection_model, segmentation_model):
    """
    Using model and tradition cv method to get background information
    :param image_path
    :param segmentation_model
    :param object_detection_model
    :return: zebra, lanes and traffic light information
    """
    img = cv2.imread(image_path)

    # Segment zebra crossing
    zebra_rect = get_zebra_rect(img, segmentation_model)

    # Get lane and lane mark
    lanes = lane_detect(img, standard_lane_marks)

    # Detect traffic light
    traffic_light_boxes = get_traffic_light(img, object_detection_model)

    return zebra_rect, lanes, traffic_light_boxes

