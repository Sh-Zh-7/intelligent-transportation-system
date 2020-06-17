import argparse
import warnings

from background import static_process
from categories import Car, Person

warnings.filterwarnings("ignore")

from Detection.keras_yolov4.yolo import Yolo4
import Tracking.video as video
from Tracking.DeepSort.deep_sort import preprocessing, nn_matching
from Tracking.DeepSort.deep_sort.detection import Detection
from Tracking.DeepSort.deep_sort.tracker import Tracker
from Tracking.DeepSort.tools import generate_detections as gdet
from Tracking.utils import *

from visualize import *

# Hyper parameters for tracking
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.45
min_box_area = 2500

# Track model settings
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)
model_filename = "./Tracking/DeepSort/model_data/market1501.pb"
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

# Global variables
flow_count = 0
cars_crossing_line = set()
tracker_db = {}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, default="./Videos/video-01.avi", help="position of your video")
    parser.add_argument("--input_background", type=str, default="./Videos/background_1.png", help="position")
    parser.add_argument("--output_dir", type=str, default="./Result", help="position to store each frame and result video")

    args = parser.parse_args()
    return args


def update_tracker(image, detection_model, tracker_model):
    # Start object detection
    img = Image.fromarray(image[..., ::-1])
    boxs, class_names = detection_model.detect_image(img)
    features = encoder(image, boxs)
    detections = [Detection(bbox, 1.0, feature, class_name) for bbox, feature, class_name in
                  zip(boxs, features, class_names)]
    # Run NMS
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    # Call the tracker
    tracker_model.predict()
    tracker_model.update(detections)


def get_environment(image, traffic_lights_bboxes):
    for bbox in traffic_lights_bboxes:
        image = np.array(image)
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        roi = image[y:y + h, x:x + w, :]
        color_str = get_traffic_light_color(roi)
        color = color_str_tuple_map[color_str]
        return color


def get_result(args, dataloader, save_dir):
    global flow_count
    global cars_crossing_line

    frame_id = 0
    timer = Timer()
    mkdir_if_missing(save_dir)

    # Using yolo and tradition cv to get background information
    yolo = Yolo4()
    zebra_rect, lanes, traffic_lights_bboxes = static_process(args, yolo)

    # Start tracking
    yolo.set_detection_class(["person", "car"])
    for frame in dataloader:
        if frame_id % 20 == 0:
            print("Processing frame {} ({:.2f} fps)".format(frame_id, 1. / max(1e-5, timer.average_time)))
        timer.tic()
        update_tracker(frame, yolo, tracker)
        traffic_light_color = get_environment(frame, traffic_lights_bboxes)

        # Store tracker result
        online_cars_ids = []
        online_persons_ids = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            tlwh = track.to_tlwh()
            # Avoid negative position
            tlwh[0], tlwh[1] = tlwh[0] if tlwh[0] > 0 else 0, tlwh[1] if tlwh[1] > 0 else 0
            # Ignore too small object
            if tlwh[2] * tlwh[3] > min_box_area:
                # Update mock track data base if the track is new
                if track.track_id not in tracker_db.keys():
                    if track.category == "car":
                        roi = np.array(frame)[int(tlwh[1]): int(tlwh[1] + tlwh[3]), 
                              int(tlwh[0]):int(tlwh[0] + tlwh[2]), :]
                        tracker_db[track.track_id] = Car(tlwh, roi, lanes, confidence=0)
                    else:
                        tracker_db[track.track_id] = Person(tlwh, confidence=0)
                # Get all tracks in this frame, will be used in plotting
                if track.category == "car":
                    online_cars_ids.append(track.track_id)
                    # Flow count
                    if track.track_id not in cars_crossing_line:
                        upper = tracker_db[track.track_id].bbox.y1
                        lower = tracker_db[track.track_id].bbox.y2
                        if upper <= frame.shape[0] // 2 <= lower:
                            flow_count += 1
                            cars_crossing_line.add(track.track_id)
                    # Update car object with environment
                    tracker_db[track.track_id].update(tlwh, traffic_light_color)
                elif track.category == "person":
                    online_persons_ids.append(track.track_id)
                    tracker_db[track.track_id].update(tlwh)
        # After get all the car object and person object
        # Detect whether a car not wait for peron
        for car_id in online_cars_ids:
            tracker_db[car_id].set_not_wait_for_person(tracker_db, online_persons_ids, zebra_rect)
        timer.toc()

        # Plot part
        # # Save tracking results, for UI information
        # results.append((frame_id + 1, online_tlwhs, online_ids))
        online_im = plot_video_info(np.array(frame), frame_id=frame_id, fps=1. / timer.average_time)
        online_im = plot_flow_statistics(online_im, flow_count)
        online_im = plot_cars_rect(online_im, online_cars_ids, tracker_db)
        online_im = plot_persons_rect(online_im, online_persons_ids, tracker_db)
        online_im = plot_traffic_bboxes(online_im, traffic_lights_bboxes)

        # Save plot images
        cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # # save results, for UI information
    # write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(args):
    # Output directory
    result_root = args.output_dir
    mkdir_if_missing(result_root)

    # Get Video class and frame_rate
    logging.info("Start tracking...")
    dataloader = video.Video(args.input_video)

    # Split video by frames
    frame_dir = os.path.join(result_root, "frame")
    get_result(args, dataloader, frame_dir)

    # Convert images to video
    output_video_path = os.path.join(result_root, "result.mp4")
    cmd_str = "ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}".format(os.path.join(result_root, "frame"),
                                                                              output_video_path)
    os.system(cmd_str)


if __name__ == '__main__':
    main(get_args())
