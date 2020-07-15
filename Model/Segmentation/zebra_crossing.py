from PIL import Image
import numpy as np
import cv2

class_colors = [[0, 0, 0], [255, 255, 255]]
NCLASSES = 2
HEIGHT, WIDTH = 416, 416

def get_zebra_crossing(img, model):
    origin_h, origin_w = img.shape[0], img.shape[1]
    img_arr = Image.fromarray(img).resize((WIDTH, HEIGHT))
    img = np.array(img_arr) / 255
    img = img.reshape(-1, HEIGHT, WIDTH, 3)

    prediction = model.predict(img)[0]
    prediction = prediction.reshape((HEIGHT // 2, WIDTH // 2, NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((HEIGHT // 2, WIDTH // 2, 3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:, :, 0] += ((prediction[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((prediction[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((prediction[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((origin_w, origin_h))
    return np.array(seg_img)


def get_zebra_area(mask):
    _, contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    target_box, max_area = None, None
    target_rect = None
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        area = rect[1][0] * rect[1][1]
        if max_area is None or area > max_area:
            max_area = area
            target_box = np.int0(cv2.boxPoints(rect))
            target_rect = rect
    return target_box, target_rect


