import cv2
from hyperlpr import *

def license_plate_recognize(car_img):
    """
    :param car_img
    :return: License plate and its confidence
    """
    try:
        result = HyperLPR_plate_recognition(car_img)
        if len(result[0][0].decode('utf-8')) == 7:
            return result[0][0], result[0][1]
        else:
            return "Can't recognize", 0
    except:
        # Can't recognize
        # Include bbox's size is too small(0)
        # And index error
        return "Can't recognize", 0


if __name__ == '__main__':
    src = cv2.imread("./test2.png")
    cv2.imshow("title", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(license_plate_recognize(src))
