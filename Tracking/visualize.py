import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def get_color(category):
    if category == "person":
        # Blue
        return 255, 0, 0
    elif category == "car":
        # Yellow
        return 0, 255, 255
    else:
        return 0, 0, 0


def plot_static_objects(image, traffic_lights_bbox, traffic_light_color):
    color_str_tuple_map = {"green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255), "black": (0, 0, 0)}
    image = plot_rectangle(image, "traffic light", traffic_lights_bbox, color_str_tuple_map[traffic_light_color])
    return image

def plot_cars(img, obj_id, tlwh, license_set, color):
    # Set font type and thickness
    image = Image.fromarray(img.copy())
    font = ImageFont.truetype(font="./Detection/keras_yolov4/font/SimSun.ttf",
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = 2

    # Set labels
    if license_set[obj_id][1] != 0:
        label = "{}".format(license_set[obj_id][0])
    else:
        label = "{}".format("<car>")
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    # Calculate bbox's position
    left, top, height, width = tlwh
    buttom, right = top + width, left + height

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, buttom - i],
            outline=color)
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=color)
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw

    return np.array(image)


def plot_tracking(image, tlwhs, obj_ids, obj_classes, license_set=None, frame_id=0, fps=0.):
    im = np.ascontiguousarray(np.copy(image))

    text_scale = max(1, image.shape[1] / 1600.)
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (image.shape[0] - 300, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        obj_class = obj_classes[i]
        obj_id = obj_ids[i]
        if obj_class == "car":
            im = plot_cars(im, obj_id, tlwh, license_set, (0, 255, 255))
        else:
            im = plot_rectangle(im, obj_class, tlwh, get_color(obj_classes[i]))
    return im


def plot_rectangle(img, obj_class, tlwh, color):
    # Set font type and thickness
    image = Image.fromarray(img.copy())
    font = ImageFont.truetype(font="./Detection/keras_yolov4/font/SimSun.ttf",
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = 2

    # Set labels
    label = "{}".format(obj_class)
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    # Calculate bbox's position
    left, top, height, width = tlwh
    buttom, right = top + width, left + height

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, buttom - i],
            outline=color)
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=color)
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw

    return np.array(image)



