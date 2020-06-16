import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from Detection.keras_yolov4.yolo import Yolo4
from Detection.traffic_light import get_traffic_light_color

import Tracking.video as video
from Tracking.DeepSort.deep_sort import preprocessing, nn_matching
from Tracking.DeepSort.deep_sort.detection import Detection
from Tracking.DeepSort.deep_sort.tracker import Tracker
from Tracking.DeepSort.tools import generate_detections as gdet
from Tracking.utils import *
from Tracking.visualize import *

import LPR.lpr as lpr

result_root = "./Result"

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.45

min_box_area = 2500

# Store id and [license_content, confident]
license_set = {}
global_count = 0
global_ids = set()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, default="./Videos/video-01.avi", help="position of your video")
    parser.add_argument("--input_background", type=str, default="./Videos/background_1.png", help="position")

    args = parser.parse_args()
    return args

def get_traffic_statistic(img, obj_id, obj_category, obj_tlwh):
    target_class = ["car"]

    global global_count, global_ids
    if obj_category in target_class and obj_id not in global_ids:
        upper, lower = int(obj_tlwh[1]), int(obj_tlwh[1] + obj_tlwh[3])
        if upper < img.shape[0] // 2 < lower:
            global_count += 1
            global_ids.add(obj_id)


def further_process(img, obj_id, obj_category, obj_tlwh):
    """
    Further process every object of every frame due to
    different task
    :param img: origin frame tracked from video
    :param obj_id: recognize same object in tracking algorithm
    :param obj_category:
    :param obj_tlwh: tl is top left coordinate(x, y), wh is width and height
    """
    # License plate recognize
    if obj_category == "car":
        roi = img[
              int(obj_tlwh[1]): int(obj_tlwh[1] + obj_tlwh[3]),
              int(obj_tlwh[0]):int(obj_tlwh[0] + obj_tlwh[2]),
              :]
        license_content, confidence = lpr.license_plate_recognize(roi)
        if obj_id in license_set:
            if confidence > license_set[obj_id][1]:
                license_set[obj_id][1] = confidence
                license_set[obj_id][0] = license_content
        else:
            license_set[obj_id] = [license_content, confidence]
    get_traffic_statistic(img, obj_id, obj_category, obj_tlwh)


def static_process(args, model):
    img = cv2.imread(args.input_background)

    model.set_detection_class(["traffic light"])
    traffic_light_box, _ = model.detect_image(img)
    return traffic_light_box


def get_result(args, dataloader, save_dir):
    global global_count

    mkdir_if_missing(save_dir)

    # --------------------------------------------Detection---------------------------------------------------
    yolo = Yolo4()
    traffic_lights_bboxes = static_process(args, yolo)

    yolo.set_detection_class(["person", "car"])
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    timer = Timer()
    frame_id = 0

    model_filename = "./Tracking/DeepSort/model_data/market1501.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # Deal with every frame
    for path, frame, img in dataloader:
        # Print PFS information
        if frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # -----------------------------------------TRACKING----------------------------------------------------
        timer.tic()
        # Start object detection
        img = Image.fromarray(frame[..., ::-1])
        boxs, class_names = yolo.detect_image(img)
        features = encoder(frame, boxs)
        detections = [Detection(bbox, 1.0, feature, class_name) for bbox, feature, class_name in
                      zip(boxs, features, class_names)]
        # Run NMS
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        # Store tracker result
        online_ids = []
        online_classes = []
        online_tlwhs = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            tlwh = track.to_tlwh()
            # Avoid negative
            tlwh[0], tlwh[1] = tlwh[0] if tlwh[0] > 0 else 0, tlwh[1] if tlwh[1] > 0 else 0
            # Ignore too small object
            if tlwh[2] * tlwh[3] > min_box_area:
                # Set min box area, otherwise the lpr won't recognize it
                online_ids.append(int(track.track_id))
                online_classes.append(track.category)
                online_tlwhs.append(tuple(tlwh))
                further_process(np.array(frame), int(track.track_id), track.category, tuple(tlwh))
        timer.toc()

        # -------------------------------------------PLOT----------------------------------------------------
        # Save tracking results
        # results.append((frame_id + 1, online_tlwhs, online_ids))
        frame = cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), color=(0, 0, 255), thickness=3)
        cv2.putText(frame, "Car counts: {}".format(global_count),
                    (frame.shape[1] // 2, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),
                    thickness=3
                    )

        online_im = plot_tracking(np.array(frame), online_tlwhs, online_ids, online_classes,
                                  license_set=license_set,
                                  frame_id=frame_id,
                                  fps=1. / timer.average_time)

        # Traffic light result
        for bbox in traffic_lights_bboxes:
            img = np.array(online_im)  # Convert to numpy object
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            roi = img[y:y + h, x:x + w, :]
            color = get_traffic_light_color(roi)
            online_im = plot_static_objects(online_im, traffic_lights_bbox=bbox, traffic_light_color=color)

        # Save plot images
        cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        frame_id += 1
    # save results
    # write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(args):
    # Output directory
    mkdir_if_missing(result_root)

    # Get Video class and frame_rate
    logging.info("Start tracking...")
    dataloader = video.Video(args.input_video)

    # Split video by frames
    frame_dir = os.path.join(result_root, "frame")
    get_result(args, dataloader, frame_dir)

    # Convert images to video
    output_video_path = os.path.join(result_root, "result.mp4")
    cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(os.path.join(result_root, 'frame'),
                                                                              output_video_path)
    os.system(cmd_str)


if __name__ == '__main__':
    main(get_args())
