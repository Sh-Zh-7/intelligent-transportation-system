import cv2
import numpy as np
from Model.Detection.keras_yolov4.yolo import Yolo4
import Model.Tracking.video as video
from PIL import Image
import matplotlib.pyplot as plt

def get_white_pixel_count(img):
    """
    Image with the most white pixels is corresponding color
    :param img: target image
    :return: white pixels count
    """
    count = 0
    for line in img:
        for element in line:
            if element != 0:
                count += 1
    return count

def get_traffic_light_color(img):
    """
    :param img: traffic light roi
    :return: "green", "red" or "yellow"
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Judge green
    green_lower = np.array([35, 43, 46])
    green_upper = np.array([77, 255, 255])
    green_mask = cv2.inRange(hsv, lowerb=green_lower, upperb=green_upper)
    cyan_lower = np.array([78, 43, 46])
    cyan_upper = np.array([99, 255, 255])
    cyan_mask = cv2.inRange(hsv, lowerb=cyan_lower, upperb=cyan_upper)
    green_count = get_white_pixel_count(green_mask + cyan_mask)

    # Judge red
    red1_lower = np.array([0, 43, 46])
    red1_upper = np.array([10, 255, 255])
    red1_mask = cv2.inRange(hsv, lowerb=red1_lower, upperb=red1_upper)
    red2_lower = np.array([156, 43, 46])
    red2_upper = np.array([180, 255, 255])
    red2_mask = cv2.inRange(hsv, lowerb=red2_lower, upperb=red2_upper)
    red_count = get_white_pixel_count(red1_mask + red2_mask)

    # Judge yellow
    orange_lower = np.array([11, 43, 46])
    orange_upper = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lowerb=orange_lower, upperb=orange_upper)
    yellow_lower = np.array([26, 43, 46])
    yellow_upper = np.array([34, 255, 255])
    yellow_mask = cv2.inRange(hsv, lowerb=yellow_lower, upperb=yellow_upper)
    yellow_count = get_white_pixel_count(orange_mask + yellow_mask)

    # Decide colors
    ret_color, max_count = "black", 0
    colors, counts = ("green", "red", "yellow"), (green_count, red_count, yellow_count)
    for color, count in zip(colors, counts):
        if count > max_count:
            ret_color = color
            max_count = count
    return ret_color


if __name__ == '__main__':
    dataloader = video.Video("../data/video-01.avi")
    # Image with static background
    img = dataloader.get_one_frame()
    img = Image.fromarray(img[..., ::-1])

    # Start detection traffic light
    yolo = Yolo4(["traffic light"])
    bboxes, _ = yolo.detect_image(img)

    for bbox in bboxes:
        img = np.array(img)     # Convert to numpy object
        x1, x2, y1, y2 = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
        roi = img[x1:x2, y1:y2, :]
        plt.imshow(roi)
        plt.show()
        print(get_traffic_light_color(roi))


