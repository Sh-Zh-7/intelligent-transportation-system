import cv2
from PIL import Image

from Segmentation.zebra_crossing import get_zebra_crossing, get_zebra_area
from Tradition.lane_detect import lane_detect, get_standard_lane_marks

standard_lane_marks = get_standard_lane_marks("./Tradition/standard_lane_marks/")


def get_zebra_rect(img):
    result = get_zebra_crossing(img)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    zebra_box, zebra_rect = get_zebra_area(gray)
    return zebra_rect


def get_traffic_light(img, model):
    img = Image.fromarray(img[..., ::-1])
    model.set_detection_class(["traffic light"])
    traffic_light_boxes, _ = model.detect_image(img)
    return traffic_light_boxes


def static_process(args, model):
    img = cv2.imread(args.input_background)

    # Segment zebra crossing
    # 这是一个List的list，内部的每一个List就是一个坐标
    zebra_rect = get_zebra_rect(img)

    # Get lane and lane mark
    lanes = lane_detect(img, standard_lane_marks)

    # Detect traffic light
    traffic_light_boxes = get_traffic_light(img, model)

    return zebra_rect, lanes, traffic_light_boxes

