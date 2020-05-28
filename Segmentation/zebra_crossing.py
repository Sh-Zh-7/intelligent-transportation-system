from Segmentation.segnet_mobile.segnet import mobilenet_segnet
from PIL import Image
import numpy as np
import cv2

class_colors = [[0, 0, 0], [255, 255, 255]]
NCLASSES = 2
HEIGHT, WIDTH = 416, 416

model = mobilenet_segnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
model.load_weights("./segnet_mobile/weights.h5")

def get_zebra_crossing(img):
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
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        area = rect[1][0] * rect[1][1]
        if max_area is None or area > max_area:
            max_area = area
            target_box = np.int0(cv2.boxPoints(rect))

    return target_box


if __name__ == '__main__':
    src = cv2.imread("./test2.png")
    result = get_zebra_crossing(src)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    target_box = get_zebra_area(gray)

    cv2.drawContours(src, [target_box], 0, (0, 0, 255), 2)
    cv2.imshow("fuck", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # result = Image.blend(Image.fromarray(src), Image.fromarray(result), 0.3)
    # cv2.imshow("title", np.array(result))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

