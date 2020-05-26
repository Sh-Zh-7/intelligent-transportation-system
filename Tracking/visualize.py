import cv2
import numpy as np


def get_color(category):
    # idx = idx * 3
    # color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    if category == "person":
        # Blue
        return 255, 0, 0
    elif category == "car":
        # Yellow
        return 0, 255, 255


def plot_static_objects(image, traffic_lights_bbox, traffic_light_color):
    color_str_tuple_map = {"green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}
    cv2.rectangle(
        image,
        (traffic_lights_bbox[0], traffic_lights_bbox[1]),
        (traffic_lights_bbox[0] + traffic_lights_bbox[2], traffic_lights_bbox[1] + traffic_lights_bbox[3]),
        color=color_str_tuple_map[traffic_light_color],
        thickness=1
    )
    cv2.putText(image, "traffic light", (int(traffic_lights_bbox[0]), int(traffic_lights_bbox[3]) + 30),
                cv2.FONT_HERSHEY_PLAIN, 1, (225, 255, 255), thickness=1)
    return image


def plot_tracking(image, tlwhs, obj_ids, obj_classes, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        obj_class = obj_classes[i]
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(obj_class)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, obj_class, (intbox[0], intbox[3] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (225, 255, 255),
                    thickness=text_thickness)
    return im
