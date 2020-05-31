import queue
import cv2


class CarDir:
    STOP = 0
    FORWORD = 1
    LEFT = 2
    RIGHT = 3


class Track:
    def __init__(self, obj_id, category, tlwh, confidence):
        self.obj_id = obj_id
        self.category = category
        self.tlwh = tlwh
        self.confidence = confidence

        self.x1, self.y1, self.w, self.h = self.tlwh
        self.x2, self.y2 = self.x1 + self.w, self.y1 + self.h
        self.center_x_numpy = int(self.y1 + self.h / 2)
        self.center_y_numpy = int(self.x1 + self.w / 2)
        self.center_numpy = [self.center_x_numpy, self.center_y_numpy]


class Car(Track):
    def __init__(self, obj_id, tlwh, confidence):
        super().__init__(obj_id, "car", tlwh, confidence)

        self.license_plate = ""
        self.license_confidence = ""

        self.belong_lane = None
        self.allow_direction = None

        # self.speed = 0    # TODO
        self.is_moving = False
        self.direction = CarDir.STOP
        self.history_center = queue.Queue(maxsize=10)

        self.is_crossing_line = False
        self.not_wait_for_person = False
        self.drive_without_guidance = False
        self.run_the_red_light = False

    def set_allow_direction(self, lanes):
        if self.allow_direction is None:
            for lane in lanes:
                if is_point_in_quad(self.center_numpy, lane.boundary):
                    self.belong_lane = lane
                    if lane.category == "left":
                        self.allow_direction = [CarDir.LEFT]
                    elif lane.category == "right":
                        self.allow_direction = [CarDir.RIGHT]
                    elif lane.category == "straight":
                        self.allow_direction = [CarDir.FORWORD]
                    elif lane.category == "straight_left":
                        self.allow_direction = [CarDir.LEFT, CarDir.FORWORD]
                    elif lane.category == "straight_right":
                        self.allow_direction = [CarDir.RIGHT, CarDir.FORWORD]
                    break

    def set_crossing_line(self):
        left_top, right_top = self.belong_lane.boundary.left_top, self.belong_lane.boundary.right_top
        x1, y1 = left_top
        x2, y2 = right_top

        k = (float(y2) - float(y1)) / (float(x2) - float(x1))
        m = (float(x2) * float(y1) - float(x1) * float(y2)) / (float(x2) - float(x1))

        self.is_crossing_line = self.center_y_numpy < self.center_x_numpy * k + m

    def


# def set_car_not_wait_for_person(car, person):
#     if person.on_crossing and car.is_moving:




class Person(Track):
    def __init__(self, obj_id, tlwh, confidence):
        super().__init__(obj_id, "person", tlwh, confidence)
        self.on_crossing = False

    def set_on_crossing(self, zebra_rect):
        person_rect = ((self.center_x_numpy, self.center_y_numpy), (self.w, self.h), 0)
        intersection = cv2.rotatedRectangleIntersection(person_rect, zebra_rect)
        self.on_crossing = (intersection != cv2.INTERSECT_NONE)


def is_point_in_quad(point, quad):
    # Using cross product to see if the point are in the same direction of lines
    x, y = point

    x1, y1 = quad.left_top
    x2, y2 = quad.right_top
    x3, y3 = quad.right_bottom
    x4, y4 = quad.left_bottom

    a = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    b = (x3 - x2) * (y - y2) - (y3 - y2) * (x - x2)
    c = (x4 - x3) * (y - y3) - (y4 - y3) * (x - x3)
    d = (x1 - x4) * (y - y4) - (y1 - y4) * (x - x4)

    if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0):
        return True
    else:
        return False
