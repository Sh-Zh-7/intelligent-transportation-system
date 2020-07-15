"""
"""
import argparse
import os

from Model.Tracking.DeepSort.deep_sort import preprocessing
from Model.Tracking.DeepSort.deep_sort.detection import Detection

from Model.visualize import *
from Model.categories import Car, Person
from Model.main import tracker_db, pts, min_box_area

# Global variables
flow_count = 0
cars_crossing_line = set()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, default="./data/video-01.avi", help="position of your video")
    parser.add_argument("--input_background", type=str, default="./data/background_1.png", help="position")
    parser.add_argument("--output_dir", type=str, default="./result",
                        help="position to store each frame and result video")

    args = parser.parse_args()
    return args


def update_tracker(image, detection_model, encoder, tracker_model, nms_max_overlap):
    # Start object detection
    img = Image.fromarray(image[..., ::-1])
    boxs, class_names, confidence = detection_model.detect_image(img)
    features = encoder(image, boxs)
    detections = [Detection(bbox, confidence, feature, class_name) for bbox, feature, class_name in
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

def process_frame(
        frame,
        tracker,
        lanes,
        all_car_info,
        traffic_light_color,
        online_cars_ids,
        online_persons_ids,
        zebra_rect,
        traffic_lights_bboxes,
        frame_id,
        timer,
        save_dir,
        fps
):
    global flow_count
    global cars_crossing_line

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
                tracker_db[track.track_id].update(tlwh, traffic_light_color, fps=fps)
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
        all_car_info += (str(frame_id / fps) + "," + str(tracker_db[car_id]))
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

