import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import warnings
warnings.filterwarnings("ignore")

from PIL import Image

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
nms_max_overlap = 0.3


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, default="./Videos/video-01.avi", help="position of your video")

    args = parser.parse_args()
    return args


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
              obj_tlwh[0]:obj_tlwh[0] + obj_tlwh[2],
              obj_tlwh[1]:obj_tlwh[1] + obj_tlwh[3],
              :]
        license_plate = lpr.license_plate_recognize(roi)

def static_process(video):
    img = video.get_one_frame()
    img = Image.fromarray(img[..., ::-1])

    yolo = Yolo4(["traffic light"])
    bboxes, _ = yolo.detect_image(img)
    return bboxes


def get_result(dataloader, save_dir):
    mkdir_if_missing(save_dir)

    # Static jobs: traffic light detection
    traffic_lights_bboxes = static_process(dataloader)

    yolo = Yolo4(["person", "car"])
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    timer = Timer()
    results = []
    frame_id = 0

    model_filename = "./Tracking/DeepSort/model_data/market1501.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # Deal with every frame
    for path, frame, img in dataloader:
        if frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        timer.tic()
        # Start object detection
        img = Image.fromarray(frame[..., ::-1])
        boxs, class_names = yolo.detect_image(img)
        features = encoder(frame, boxs)
        detections = [Detection(bbox, 1.0, feature, class_name) for bbox, feature, class_name in zip(boxs, features, class_names)]
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
            online_ids.append(int(track.track_id))
            online_classes.append(track.category)
            online_tlwhs.append(tuple(track.to_tlwh()))

            # further_process(np.array(frame), int(track.track_id), track.category, tuple(track.to_tlwh()))
        timer.toc()

        # Traffic light result
        for bbox in traffic_lights_bboxes:
            img = np.array(frame)  # Convert to numpy object
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            roi = img[y:y + h, x:x + w, :]
            color = get_traffic_light_color(roi)
            frame = plot_static_objects(frame, traffic_lights_bbox=bbox, traffic_light_color=color)

        # Save tracking results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        online_im = plot_tracking(np.array(frame), online_tlwhs, online_ids, online_classes, frame_id=frame_id,
                                  fps=1. / timer.average_time)

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
    get_result(dataloader, frame_dir)

    # Convert images to video
    output_video_path = os.path.join(result_root, "result.mp4")
    cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(os.path.join(result_root, 'frame'),
                                                                              output_video_path)
    os.system(cmd_str)


if __name__ == '__main__':
    main(get_args())
