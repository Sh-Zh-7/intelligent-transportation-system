# Python standard lib
import json
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
from background import static_process
from utils import *

# Hyper parameters for tracking
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.45
min_box_area = 2500

# Global variable
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
    command = "ffprobe -v error -select_streams v -show_entries stream=r_frame_rate -of json {}".format(video_path)
    value = os.popen(command).read()
    data = json.loads(value)
    frame_rate = eval(data.get('streams')[0].get('r_frame_rate'))
    return frame_rate


def get_result(video_path, image_path, output_dir, models, fps):
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

        frame_id, all_car_info = process_frame(
            frame, tracker, lanes,
            traffic_light_color, online_cars_ids,
            online_persons_ids, zebra_rect, traffic_lights_bboxes,
            frame_id, timer, save_dir, fps)
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
    fps = get_video_fps(video_path)
    get_result(video_path, image_path, output_dir, models, fps)
    
