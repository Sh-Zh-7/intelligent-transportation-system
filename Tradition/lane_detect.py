import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_lines(src):
    height, width = src.shape
    # Select lower half part of origin image as our roi
    roi = src[height // 2:height, :]

    dst = cv2.Canny(roi, threshold1=50, threshold2=100)
    cv2.imshow("fuck", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines = cv2.HoughLines(dst, rho=1, theta=np.pi / 180, threshold=200)
    lines_in_plor = lines[:, 0, :]

    return lines_in_plor

def transform_polar_2_rect(polar_lines):
    rect_lines = []
    for line in polar_lines:
        rho, theta = line

        a, b = np.cos(theta), np.sin(theta)
        # Center
        x0, y0 = a * rho, b * rho
        # Start and end
        x1, y1 = int(x0 + 10000 * (-b)), int(y0 + 10000 * a)
        x2, y2 = int(x0 - 10000 * (-b)), int(y0 - 10000 * a)
        rect_lines.append([(x1, y1), (x2, y2)])

    return rect_lines


def translate_polar(polar_line, distance):
    old_rho, theta = polar_line
    new_rho = old_rho + distance * np.sin(theta)
    return new_rho, theta


def get_3_lines(lines):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(lines)
    return kmeans.cluster_centers_


def plot_lines_on_img(img, lines):
    for line in lines:
        cv2.line(img, line[0], line[1], color=(0, 0, 255), thickness=3)


def two_point_2_km(two_point_line):
    point1, point2 = two_point_line
    x1, y1 = point1
    x2, y2 = point2

    k = (y2 - y1) / (x2 - x1)
    m = (x2 * y1 - x1 * y2) / (x2 - x1)
    return k, m


def get_intersection(line1, line2):
    k1, m1 = line1
    k2, m2 = line2

    x = (m2 - m1) / (k1 - k2)
    y = (k1 * m2 - k2 * m1) / (k1 - k2)
    return int(x), int(y)


def find_horizontal(lines):
    ret_line = lines[0]
    k_min = abs(ret_line[0])

    for line in lines:
        if abs(line[0]) < k_min:
            ret_line = line
            k_min = abs(line[0])

    lines.remove(ret_line)
    return ret_line, lines


if __name__ == '__main__':
    # Line detection by Hough transform
    src = cv2.imread("test2.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)[1]
    blur = cv2.GaussianBlur(binary, (7, 7), 0)
    kernel = np.ones((5, 5), np.uint8)
    dst = cv2.dilate(blur, kernel)
    lines = get_lines(dst)

    # Clustering
    old_three_lines = get_3_lines(lines)
    new_three_lines = []
    distance = src.shape[0] // 2
    for line in lines:
        new_three_lines.append(translate_polar(line, distance))

    # # Redundant
    # three_lines_in_two_points = transform_polar_2_rect(new_three_lines)
    # three_lines_in_km = []
    # for line in three_lines_in_two_points:
    #     three_lines_in_km.append(two_point_2_km(line))
    # horizontal_line, other_lines = find_horizontal(three_lines_in_km)
    #
    # # Get 4 point to decide the lane regine
    # top_point1 = get_intersection(horizontal_line, other_lines[0])
    # top_point2 = get_intersection(horizontal_line, other_lines[1])
    # bottom_point1 = get_intersection((0, src.shape[0]), other_lines[0])
    # bottom_point2 = get_intersection((0, src.shape[0]), other_lines[1])
    # if top_point1[0] < top_point2[0]:
    #     top_left_point = top_point1
    #     top_right_point = top_point2
    # else:
    #     top_left_point = top_point2
    #     top_right_point = top_point1
    # if bottom_point1[0] < bottom_point2[0]:
    #     bottom_left_point = bottom_point1
    #     bottom_right_point = bottom_point2
    # else:
    #     bottom_left_point = bottom_point2
    #     bottom_right_point = bottom_point1
    #
    # cv2.line(src, top_left_point, bottom_left_point, color=(0, 0, 255), thickness=3)
    # cv2.line(src, top_left_point, top_right_point, color=(0, 0, 255), thickness=3)
    # cv2.line(src, top_right_point, bottom_right_point, color=(0, 0, 255), thickness=3)

    # Plotting
    plot_lines_on_img(src, transform_polar_2_rect(new_three_lines))

    cv2.imshow("fuck", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

