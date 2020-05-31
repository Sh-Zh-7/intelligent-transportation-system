import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.measure import label, compare_mse
from PIL import Image

# # create sift
# sift = cv2.xfeatures2d.SIFT_create()
# # create FLANN match template
# FLANN_INDEX_KDTREE = 0
# indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# searchParams = dict(checks=50)
# flann = cv2.FlannBasedMatcher(indexParams, searchParams)

def show_img(src):
    cv2.imshow("title", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def largest_connect_component(bw_img):
    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)

    max_label = 0
    max_num = 0
    # Start with 1 to avoid background
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc


def contour_area_threshold(contours, threshold):
    ret_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= threshold:
            ret_contours.append(contour)

    return ret_contours


def transform_by_kmeans(points,  n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    return kmeans.cluster_centers_


def get_irregular_roi(img, seed):
    mask = np.zeros([img.shape[0] + 2, img.shape[1] + 2], np.uint8)
    cv2.floodFill(img, mask=mask, seedPoint=seed, newVal=255)


def find_largest_bbox(contours):
    max_area = -1
    ret_index = 0
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            ret_index = index
    return contours[ret_index]


def getMatchNum(matches, ratio):
    matchesMask = [[0, 0]] * len(matches)
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            matchesMask[i] = [1, 0]
            matchNum += 1
    return matchNum, matchesMask


# def cal_sim_sift(img1, img2):
#     img1 = np.array(Image.fromarray(img1).resize(reversed(img2.shape)))
#     img1, img2 = np.uint8(img1), np.uint8(img2)
#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)
#     matches = flann.knnMatch(des1, des2, k=2)
#     matchNum, matchesMask = getMatchNum(matches, 0.95)
#     matchRatio = matchNum * 100 / len(matches)
#     # print(matchRatio)
#     return matchRatio


def cal_sim_contours(img1, img2):
    img1 = np.array(Image.fromarray(img1).resize(reversed(img2.shape)))
    contours1 = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2 = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sim = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I2, 0)
    # print(sim)
    return sim


def cal_sim_sse(img1, img2):
    img1 = np.array(Image.fromarray(img1).resize(reversed(img2.shape)))
    score = compare_mse(img1, img2)
    # print(score)
    return score

