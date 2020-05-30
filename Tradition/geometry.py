import cv2

class QuadPoints:
    """
    Lane's 4 corner points
    """
    def __init__(self, point_list):
        # Use y-axis to classify upper and lower points
        sorted_point_list = sorted(point_list, reverse=True, key=lambda x: x[1])

        top_points = sorted_point_list[:2]
        if top_points[0][0] < top_points[1][0]:
            self.left_top = tuple(top_points[0])
            self.right_top = tuple(top_points[1])
        else:
            self.left_top = tuple(top_points[1])
            self.right_top = tuple(top_points[0])

        bottom_points = sorted_point_list[2:]
        if bottom_points[0][0] < bottom_points[1][0]:
            self.left_bottom = tuple(bottom_points[0])
            self.right_bottom = tuple(bottom_points[1])
        else:
            self.left_bottom = tuple(bottom_points[1])
            self.right_bottom = tuple(bottom_points[0])

    def plot_on_images(self, img, color=(0, 255, 0), thickness=3):
        cv2.line(img, self.left_top, self.right_top, color=color, thickness=thickness)
        cv2.line(img, self.right_top, self.right_bottom, color=color, thickness=thickness)
        cv2.line(img, self.right_bottom, self.left_bottom, color=color, thickness=thickness)
        cv2.line(img, self.left_bottom, self.left_top, color=color, thickness=thickness)

    def get_center(self):
        line1 = Line([self.left_top, self.right_bottom], mode=LineMode.TWO_POINTS)
        line2 = Line([self.right_top, self.left_bottom], mode=LineMode.TWO_POINTS)
        return get_intersection(line1, line2)

class LineMode:
    """
    Define line's mode
    """
    POLAR = 0
    RECT = 1
    TWO_POINTS = 2

class Line:
    """
    2D line's info, store by km
    """
    def __init__(self, info, mode):
        if mode == LineMode.POLAR:
            pass
        elif mode == LineMode.RECT:
            self.k, self.m = info
        elif mode == LineMode.TWO_POINTS:
            point1, point2 = info
            x1, y1 = point1
            x2, y2 = point2

            # Avoid uint overflow
            self.k = (float(y2) - float(y1)) / (float(x2) - float(x1))
            self.m = (float(x2) * float(y1) - float(x1) * float(y2)) / (float(x2) - float(x1))


def get_intersection(line1, line2):
    x = (line2.m - line1.m) / (line1.k - line2.k)
    y = (line1.k * line2.m - line2.k * line1.m) / (line1.k - line2.k)
    return int(x), int(y)
