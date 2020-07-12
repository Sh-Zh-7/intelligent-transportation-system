# Python standard lib
import json
import subprocess
import warnings
from collections import deque

warnings.filterwarnings("ignore")

# Detection import
from Detection.keras_yolov4.yolo import Yolo4
# Tracking import
import Tracking.video as video
from Tracking.DeepSort.deep_sort import nn_matching
from Tracking.DeepSort.deep_sort.tracker import Tracker
from Tracking.DeepSort.tools import generate_detections as gdet
from Tracking.utils import *
# Segmentation import
from Segmentation.segnet_mobile.segnet import mobilenet_segnet
from Segmentation.zebra_crossing import WIDTH, HEIGHT, NCLASSES

# Utils
from visualize import *
from background import static_process
from categories import Car, Person
from utils import *

# Hyper parameters for tracking
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.45
min_box_area = 2500

# Global variables
flow_count = 0
cars_crossing_line = set()
tracker_db = {}
# Cars center set
pts = [deque(maxlen=30) for _ in range(9999)]


def load_models():
    # YOLO v4
    yolo = Yolo4()
    # Deep sort
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    model_filename = "./Tracking/DeepSort/model_data/market1501.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # Mobile net and segnet
    ms_model = mobilenet_segnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
    ms_model.load_weights("./Segmentation/segnet_mobile/weights.h5")

    return yolo, tracker, encoder, ms_model


def get_video_fps(video_path):
    command = "ffprobe  -v error -select_streams v -show_entries stream=r_frame_rate -of json {}".format(video_path)
    value = subprocess.check_output(command)
    data = json.loads(value)
    frame_rate = eval(data.get('streams')[0].get('r_frame_rate'))
    return frame_rate


def process_frame(
        frame,
        tracker,
        lanes,
        traffic_light_color,
        online_cars_ids,
        online_persons_ids,
        zebra_rect,
        traffic_lights_bboxes,
        frame_id,
        timer,
        save_dir
):
    global flow_count
    global cars_crossing_line

    all_car_info = ""
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
                    tracker_db[track.track_id] = Car(track.track_id, tlwh, roi, lanes, confidence=track.confidence)
                else:
                    tracker_db[track.track_id] = Person(track.track_id, tlwh, confidence=track.confidence)
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
                # Update cars center set to plot motion path
                # We cannot iterate the queue, so we just draw the line here
                pts[track.track_id].append(tracker_db[track.track_id].bbox.center)
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None \
                            or pts[track.track_id][j - 1] == pts[track.track_id][j]:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame, tuple(pts[track.track_id][j - 1]), tuple(pts[track.track_id][j]),
                             (0, 0, 255), thickness)
            elif track.category == "person":
                online_persons_ids.append(track.track_id)
                tracker_db[track.track_id].update(tlwh)
    # After get all the car object and person object
    # Detect whether a car not wait for peron
    for car_id in online_cars_ids:
        tracker_db[car_id].set_not_wait_for_person(tracker_db, online_persons_ids, zebra_rect)
        # Saving tracking results, for UI information
        all_car_info += (str(frame_id) + "," + str(tracker_db[car_id]))
    timer.toc()

    # Plot part
    online_im = plot_video_info(np.array(frame), frame_id=frame_id, fps=1. / timer.average_time)
    online_im = plot_flow_statistics(online_im, flow_count)
    online_im = plot_cars_rect(online_im, online_cars_ids, tracker_db)
    online_im = plot_persons_rect(online_im, online_persons_ids, tracker_db)
    online_im = plot_traffic_bboxes(online_im, traffic_lights_bboxes)

    # Save plot images
    cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
    frame_id += 1
    return frame_id, all_car_info


def get_result(video_path, image_path, output_dir, models):
    # Output directory
    result_root = output_dir
    mkdir_if_missing(result_root)
    # Load video
    dataloader = video.Video(video_path)
    save_dir = os.path.join(output_dir, "frame")
    # Other initialize
    frame_id = 0
    timer = Timer()
    all_car_info = ""
    mkdir_if_missing(save_dir)

    # Unpacking the models
    yolo, tracker, encoder, ms_model = models

    # Using yolo and tradition cv to get background information
    zebra_rect, lanes, traffic_lights_bboxes = static_process(image_path, yolo, ms_model)

    # Start tracking
    yolo.set_detection_class(["person", "car"])
    for frame in dataloader:
        if frame_id % 20 == 0:
            print("Processing frame {} ({:.2f} fps)".format(frame_id, 1. / max(1e-5, timer.average_time)))
        timer.tic()
        update_tracker(frame, yolo, encoder, tracker, nms_max_overlap)
        traffic_light_color = get_environment(frame, traffic_lights_bboxes)

        # Store tracker result
        online_cars_ids = []
        online_persons_ids = []

        frame_id, all_car_info = process_frame(frame, tracker, lanes, traffic_light_color, online_cars_ids,
                                     online_persons_ids, zebra_rect, traffic_lights_bboxes, frame_id, timer, save_dir)
        # Add your code here...
        print(frame_id)

    # Dump it into .txt file
    with open(os.path.join(output_dir, "result.txt"), "w") as f:
        f.write(all_car_info)

    # Convert images to video
    output_video_path = os.path.join(output_dir, "result.mp4")
    cmd_str = "ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}".format(os.path.join(output_dir, "frame"),
                                                                              output_video_path)
    os.system(cmd_str)


if __name__ == '__main__':
    # Introduce how to use
    args = get_args()
    # Load models
    models = load_models()

    # Get Video class and frame_rate
    video_path = args.input_video
    image_path = args.input_background
    output_dir = args.output_dir
    # Get result
    # fps = get_video_fps(video_path)
    get_result(video_path, image_path, output_dir, models)
    
