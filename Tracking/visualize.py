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


def plot_static_objects(image, traffic_lights_bbox, traffic_light_color):
    color_str_tuple_map = {"green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}
    image = plot_rectangle(image, "traffic light", traffic_lights_bbox, color_str_tuple_map[traffic_light_color])
    return image


def plot_tracking(image, tlwhs, obj_ids, obj_classes, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))

    text_scale = max(1, image.shape[1] / 1600.)
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        im = plot_rectangle(im, obj_classes[i], tlwh, get_color(obj_classes[i]))
    return im


def plot_rectangle(img, obj_class, tlwh, color):
    # Set font type and thickness
    image = Image.fromarray(img.copy())
    font = ImageFont.truetype(font="./Detection/keras_yolov4/font/FiraMono-Medium.otf",
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

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



