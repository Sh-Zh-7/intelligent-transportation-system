import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from Detection.traffic_light import get_traffic_light_color

color_str_tuple_map = {"green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255), "black": (0, 0, 0)}

def plot_video_info(image, frame_id, fps):
    """
    Plot video output information on each frames
    :param image: each frame
    :param frame_id: frame id information, in order to
    :param fps: FPS, indicate the inference time
    :return: plotted frame
    """
    im = np.ascontiguousarray(np.copy(image))

    text_scale = max(1, image.shape[1] / 1600.)
    cv2.putText(im, 'frame: %d fps: %.2f' % (frame_id, fps),
                (image.shape[0] - 300, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                thickness=2)

    return im


def plot_flow_statistics(image, car_count):
    """
    Plot horizontal line on the center of image,
    put car flow count above the line
    :param image: frame of the video
    :param car_count: the number of car which across the line
    :return: frame after processing
    """
    image = cv2.line(image, (0, image.shape[0] // 2), (image.shape[1], image.shape[0] // 2), color=(0, 0, 255),
                     thickness=3)
    cv2.putText(image, "car counts: {}".format(car_count),
                (image.shape[1] // 2, image.shape[0] // 2),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                thickness=3
                )
    return image


def plot_cars_rect(image, cars_id, tracker_db, color=(0, 255, 255)):
    # Set font type and thickness
    image = Image.fromarray(image.copy())
    font = ImageFont.truetype(font="./Detection/keras_yolov4/font/SimSun.ttf",
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = 2
    for car_id in cars_id:
        car = tracker_db[car_id]
        if car.license_confidence != 0:
            label = "{}".format(car.license_plate)
        else:
            label = "{}".format("car")
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        if car.bbox.x1 - label_size[1] >= 0:
            text_origin = np.array([car.bbox.x1, car.bbox.y1 - label_size[1]])
        else:
            text_origin = np.array([car.bbox.x1, car.bbox.y1 + 1])

        for i in range(thickness):
            draw.rectangle([car.bbox.x1 + i, car.bbox.y1 + i, car.bbox.x2 - i, car.bbox.y2 - i], outline=color)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)


def plot_persons_rect(image, persons_id, tracker_db, color=(255, 0, 0)):
    # Set font type and thickness
    image = Image.fromarray(image.copy())
    font = ImageFont.truetype(font="./Detection/keras_yolov4/font/SimSun.ttf",
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = 2

    for person_id in persons_id:
        person = tracker_db[person_id]
        # Set labels
        label = "person"
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        if person.bbox.x1 - label_size[1] >= 0:
            text_origin = np.array([person.bbox.x1, person.bbox.y1 - label_size[1]])
        else:
            text_origin = np.array([person.bbox.x1, person.bbox.y1 + 1])

        for i in range(thickness):
            draw.rectangle([person.bbox.x1 + i, person.bbox.y1 + i, person.bbox.x2 - i, person.bbox.y2 - i], outline=color)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)


def plot_traffic_bboxes(image, traffic_lights_bboxes):
    image = Image.fromarray(image.copy())
    font = ImageFont.truetype(font="./Detection/keras_yolov4/font/SimSun.ttf",
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = 2
    # Traffic light result
    for bbox in traffic_lights_bboxes:
        image = np.array(image)
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        roi = image[y:y + h, x:x + w, :]
        color_str = get_traffic_light_color(roi)
        color = color_str_tuple_map[color_str]

        image = Image.fromarray(image.copy())
        # Set labels
        label = "traffic light"
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        # Calculate bbox's position
        left, top, height, width = bbox
        buttom, right = top + width, left + height

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, buttom - i], outline=color)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    return np.array(image)

