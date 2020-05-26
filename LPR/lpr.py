import cv2
from hyperlpr import *

def license_plate_recognize(car_img):
    """
    :param car_img
    :return: License plate and its confidence
    """
    try:
        result = HyperLPR_plate_recognition(car_img)
        return result[0][0], result[0][1]
    except:
        # Can't recognize
        # Include bbox's size is too small(0)
        # And index error
        return "Can't recognize", 0


if __name__ == '__main__':
    src = cv2.imread("test.jpg")
    license_plate_recognize(src)
