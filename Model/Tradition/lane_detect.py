from Model.Tradition.geometry import *
from Model.Tradition.lane_mark import *

import warnings
warnings.filterwarnings("ignore")


class Lane:
    def __init__(self, boundary, category):
        self.boundary = boundary
        self.category = category


def lane_detect(src, standard_lane_marks):
    ret_lanes = []
    height, width, _ = src.shape

    # Pre-processing
    roi = src[height // 2:height, :, :]
    # Pre-processing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_binary = cv2.threshold(gray, thresh=120, maxval=255, type=cv2.THRESH_BINARY)[1]
    roi_blur = cv2.GaussianBlur(roi_binary, (15, 15), 0)
    # Morphology operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(roi_blur, cv2.MORPH_OPEN, kernel)

    # Fill the edges by draw a lot of lines
    zero_background = np.zeros((roi.shape[0], roi.shape[1], 3))
    mask = zero_background.copy()
    lines = cv2.HoughLinesP(dst, rho=1, theta=np.pi / 180, threshold=10)
    lines_in_plar = lines[:, 0, :]
    for line in lines_in_plar:
        cv2.line(mask, tuple(line[:2]), tuple(line[2:]), color=(0, 0, 255), thickness=2)

    # Separate zebra crossing and lane by using largest connect component
    mask = np.float32(mask)
    binary = cv2.threshold(mask, thresh=127, maxval=255, type=cv2.THRESH_BINARY)[1]
    result = largest_connect_component(binary)
    result = np.float32(result)
    kernel = np.ones((15, 15), np.uint8)
    result = cv2.dilate(result, kernel)

    # Divide lanes by using largest area of contours
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY)[1]
    binary = np.uint8(binary)
    binary[roi.shape[0] - 10:roi.shape[0], :] = 255
    binary[:, 0:10] = 255
    binary[:, roi.shape[1] - 5:roi.shape[1]] = 255
    _, contours, hierarchies = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Select areas that big enough
    target_contours = []
    for contour, hierarchy in zip(contours, hierarchies[0]):
        if hierarchy[3] >= 0:
            target_contours.append(contour)
    result_contours = contour_area_threshold(target_contours, 115000)

    # DO NOT use list multiple operation
    zero_backgrounds = [np.zeros((roi.shape[0], roi.shape[1], 3)),
                        np.zeros((roi.shape[0], roi.shape[1], 3)),
                        np.zeros((roi.shape[0], roi.shape[1], 3))]
    zero_backgrounds_tmp = zero_backgrounds.copy()
    for index, contour in enumerate(result_contours):
        # Convert to convex shape
        hull_points = cv2.convexHull(contour)
        hull_points = hull_points[:, 0, :]
        # Fit the contours by poly method
        corners = cv2.approxPolyDP(hull_points, epsilon=30, closed=True)
        corners = transform_by_kmeans(corners[:, 0, :])
        corners = np.uint32(corners)
        quad_points = QuadPoints(corners)
        quad_points.plot_on_images(zero_backgrounds_tmp[index])

        # Fill the contour to get the ROI
        zero_backgrounds_tmp[index] = np.uint8(zero_backgrounds_tmp[index])
        gray = cv2.cvtColor(zero_backgrounds_tmp[index], cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        seed_point = quad_points.get_center()
        get_irregular_roi(binary, seed_point)
        mark_roi = cv2.bitwise_and(roi_blur, binary)

        # Adjust angle
        src = np.array([list(quad_points.left_top), list(quad_points.right_top),
                        list(quad_points.left_bottom), list(quad_points.right_bottom)], dtype="float32")
        dst = np.array([[0, 800], [400, 800], [0, 0], [400, 0]], dtype="float32")
        perspective_matrix = cv2.getPerspectiveTransform(src, dst)
        perspective = cv2.warpPerspective(mark_roi, perspective_matrix, (400, 800), cv2.INTER_LINEAR)

        # Get Lane class object
        lane_mark_category = recognize_lane_mark(perspective, standard_lane_marks)
        translation = height - height // 2
        quad_points.adjust_position(translation)
        lane = Lane(quad_points, lane_mark_category)
        ret_lanes.append(lane)

    return ret_lanes


if __name__ == '__main__':
    standard_lane_marks = get_standard_lane_marks("./standard_lane_marks/")
    img = cv2.imread("test.jpg")
    lanes = lane_detect(img)
    for lane in lanes:
        lane.boundary.plot_on_images(img)
        cv2.putText(img, map_int_2_direction_str(lane.category),
                    lane.boundary.get_center(), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 3)
    show_img(img)


