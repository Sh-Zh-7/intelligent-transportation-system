import os
from Model.Tradition.utils import *

class LaneMarkDirection:
    LEFT = 0
    RIGHT = 1
    STRAIGHT = 2
    STRAIGHT_LEFT = 3
    STRAIGHT_RIGHT = 4


def map_int_2_direction_str(num):
    strs = ["left", "right", "straight", "straight_left", "straight_right"]
    return strs[num]


def get_standard_lane_marks(path):
    standard_lane_marks = []
    image_files = os.listdir(path)
    image_files.sort(key=lambda x: x[0])
    for image in image_files:
        src = cv2.imread(path + image)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        standard_lane_marks.append(binary)
    return standard_lane_marks


def recognize_lane_mark(src, standard_lane_marks):
    _, contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    target_contours = find_largest_bbox(contours)

    x, y, w, h = cv2.boundingRect(target_contours)

    min_sim, target_index = None, 0
    lane_mark = src[y: y + h, x: x + w]
    for index, standard in enumerate(standard_lane_marks):
        sim = cal_sim_sse(standard, lane_mark)
        if min_sim is None or sim < min_sim:
            min_sim = sim
            target_index = index
    return target_index

