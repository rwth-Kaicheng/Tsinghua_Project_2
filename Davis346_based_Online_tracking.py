import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
import cv2 as cv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from transformers import MarianMTModel, MarianTokenizer
import spacy
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import dv_processing as dv

# Initialize models and libraries
zh2en_model = 'Helsinki-NLP/opus-mt-zh-en'
tokenizer = MarianTokenizer.from_pretrained(zh2en_model)
zh2enmodel = MarianMTModel.from_pretrained(zh2en_model)
nlp = spacy.load("en_core_web_md")

# Initialize DataFrame
columns = ['Source', 'Label', 'Score', 'X1', 'Y1', 'X2', 'Y2', 'Center_X', 'Center_Y', 'Timestamp', 'Inference_Time', 'Inference_Interval', 'ID', 'Event_Inference_Time', 'Kalman_Inference_Time', 'Total_Inference_Time', 'Kalman_Center_X', 'Kalman_Center_Y']
df = pd.DataFrame(columns=columns)

# Colors class
class Colors:
    def __init__(self):
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86",
            "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF",
            "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

# Translation function
def translate_zh_to_en(text):
    encoded_text = tokenizer(text, return_tensors="pt", padding=True)
    translated = zh2enmodel.generate(**encoded_text)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Filter the highest confidence box for each category
def filter_highest_confidence_boxes(boxes, scores, labels):
    highest_confidence_boxes = {}
    for box, score, label in zip(boxes, scores, labels):
        if label not in highest_confidence_boxes or highest_confidence_boxes[label][1] < score:
            highest_confidence_boxes[label] = (box, score)
    filtered_boxes = [v[0] for v in highest_confidence_boxes.values()]
    filtered_scores = [v[1] for v in highest_confidence_boxes.values()]
    filtered_labels = list(highest_confidence_boxes.keys())
    return filtered_boxes, filtered_scores, filtered_labels

# Image drawing function for individual frames
def draw_boxes_on_frame(frame, boxes, labels, scores, colors, ids):
    font = cv.FONT_HERSHEY_SIMPLEX
    for box, label, score, color, obj_id in zip(boxes, labels, scores, colors, ids):
        x1, y1, x2, y2 = map(int, box)
        label_text = f'{label}:{score:.2f} (ID:{obj_id})'
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, label_text, (x1, y1 - 10), font, 0.5, color, 2)
    return frame

# GLIP model configuration
config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = r'E:/premodel/glip_tiny_model_o365_goldg_cc_sbu.pth'
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,  # Increase confidence threshold
    show_mask_heatmaps=False
)

# GLIP inference function
def glip_inference(image_, caption_):
    colors_ = Colors()
    preds = glip_demo.compute_prediction(image_, caption_)
    top_preds = glip_demo._post_process(preds, threshold=0.3)
    labels = top_preds.get_field("labels").tolist()
    scores = top_preds.get_field("scores").tolist()
    boxes = top_preds.bbox.detach().cpu().numpy()
    colors = [colors_(idx) for idx in labels]
    labels_names = glip_demo.get_label_names(labels)
    return boxes, scores, labels_names, colors

# Video resolution
resolution = (346, 260)

# Initialize accumulator
accumulator = dv.Accumulator(resolution)
accumulator.setMinPotential(0.0)
accumulator.setMaxPotential(1.0)
accumulator.setNeutralPotential(0.5)
accumulator.setEventContribution(0.15)
accumulator.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
accumulator.setDecayParam(1e+7)
accumulator.setIgnorePolarity(False)
accumulator.setSynchronousDecay(False)

# Initialize slicer
slicer = dv.EventStreamSlicer()

# Initialize Kalman filter for each box
class KalmanFilter:
    def __init__(self, initial_state, initial_velocity):
        dt = 1  # Initial time step
        self.kf = cv.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 10  # Increased process noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.001  # Decreased measurement noise covariance
        self.kf.statePost = np.array([[initial_state[0]], [initial_state[1]], [initial_velocity[0]], [initial_velocity[1]]], dtype=np.float32)  # Initialize state
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1e-2  # Initialize state covariance with higher uncertainty

    def predict(self):
        prediction = self.kf.predict()[:2].reshape(-1)
        return prediction

    def correct(self, coords):
        coords = np.array([[coords[0]], [coords[1]]], dtype=np.float32)
        self.kf.correct(coords)

    def adjust_noise_covariance(self, event_based_prediction, kalman_prediction):
        # Calculate the difference between event prediction and Kalman prediction
        prediction_error = np.linalg.norm(event_based_prediction - kalman_prediction)

        # Adjust process noise covariance based on prediction error
        if prediction_error > 3.0:  # Significant prediction error threshold
            self.kf.processNoiseCov *= 1.1  # Increase process noise covariance
            self.kf.measurementNoiseCov *= 1.1  # Increase measurement noise covariance
        else:
            self.kf.processNoiseCov *= 0.9  # Decrease process noise covariance
            self.kf.measurementNoiseCov *= 0.9  # Decrease measurement noise covariance

kalman_filters = {}

# Calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Define function to update box position based on center
def update_box_position(box, new_center, max_shift=25):
    (center_x, center_y), (width, height) = box
    shift_x = new_center[0] - center_x
    shift_y = new_center[1] - center_y
    if abs(shift_x) > max_shift:
        shift_x = max_shift if shift_x > 0 else -max_shift
    if abs(shift_y) > max_shift:
        shift_y = max_shift if shift_y > 0 else -max_shift
    new_center_x = center_x + shift_x
    new_center_y = center_y + shift_y
    return np.array([[new_center_x, new_center_y], [width, height]], dtype=np.float32)

# Calculate new center based on events
def calculate_new_center(events, prev_center, width, height):
    if events is None or len(events) == 0:
        return prev_center.astype(np.float32)

    x_min = prev_center[0] - width / 2
    x_max = prev_center[0] + width / 2
    y_min = prev_center[1] - height / 2
    y_max = prev_center[1] + height / 2
    mask = (events['x'] >= x_min) & (events['x'] <= x_max) & (events['y'] >= y_min) & (events['y'] <= y_max)
    filtered_events = events[mask]

    if len(filtered_events) == 0:
        return prev_center.astype(np.float32)

    return np.array([np.mean(filtered_events['x']), np.mean(filtered_events['y'])], dtype=np.float32)

# Initialize FPS calculation
frame_count = 0
fps_start_time = time.perf_counter()
single_tracking_times = []

# Global variable to store previous distances
previous_distances = {}

# Initialize tracking state
is_tracking = False

# Declare slicing callback method
def slicing_callback(events: dv.EventStore):
    global tracked_boxes, frame_count, fps_start_time, glip_labels, start_tracking_time, tracking_duration, is_tracking, df, out, capture, single_tracking_times, object_ids, kalman_filters, previous_distances

    if not is_tracking:
        return

    if time.perf_counter() - start_tracking_time > tracking_duration:
        is_tracking = False
        out.release()
        cv.destroyAllWindows()
        return

    start_time = time.perf_counter()
    accumulator.clear()

    # Accept event data
    accumulator.accept(events)
    frame = accumulator.generateFrame().image
    _, binary_frame = cv.threshold(frame, 1, 255, cv.THRESH_BINARY)
    event_coords = np.array([(event.x(), event.y()) for event in events])

    other_centers = {object_ids[i]: tracked_boxes[i][0] for i in range(len(tracked_boxes))}
    other_boxes = {object_ids[i]: [tracked_boxes[i][0][0] - tracked_boxes[i][1][0] / 2, tracked_boxes[i][0][1] - tracked_boxes[i][1][1] / 2, tracked_boxes[i][0][0] + tracked_boxes[i][1][0] / 2, tracked_boxes[i][0][1] + tracked_boxes[i][1][1] / 2] for i in range(len(tracked_boxes))}

    for i, box in enumerate(tracked_boxes):
        tracking_start_time = time.perf_counter()
        previous_center = box[0]
        nearby_events = pd.DataFrame(event_coords, columns=['x', 'y'])
        start_event_inference_time = time.perf_counter()
        new_center_event_based = calculate_new_center(nearby_events, box[0], box[1][0], box[1][1])
        end_event_inference_time = time.perf_counter()
        event_inference_time = end_event_inference_time - start_event_inference_time

        start_kalman_inference_time = time.perf_counter()
        new_center_kalman = kalman_filters[object_ids[i]].predict()
        end_kalman_inference_time = time.perf_counter()
        kalman_inference_time = end_kalman_inference_time - start_kalman_inference_time

        # Calculate distance between event-based center and Kalman predicted center
        loss = np.linalg.norm(new_center_event_based - new_center_kalman)

        # Check distances with other boxes to determine whether to use event tracking or Kalman filter
        use_kalman = False
        for other_id, other_center in other_centers.items():
            if other_id != object_ids[i]:
                distance = calculate_distance(previous_center, other_center)
                if distance < 40:
                    use_kalman = True
                    previous_distances[(object_ids[i], other_id)] = distance
                elif (object_ids[i], other_id) in previous_distances and distance > 40:
                    use_kalman = False

        if use_kalman:
            new_center = new_center_kalman
            source = 'Kalman filter'
        else:
            kalman_filters[object_ids[i]].correct(new_center_event_based)
            new_center = new_center_event_based
            source = 'Event tracking'

        # Update detection box position
        tracked_boxes[i] = update_box_position(box, new_center)
        center = tracked_boxes[i][0]

        timestamp = int(datetime.now().timestamp() * 1e6)
        total_inference_time = event_inference_time + kalman_inference_time
        df.loc[len(df)] = ['event', glip_labels[i], None, box[0][0] - box[1][0] / 2, box[0][1] - box[1][1] / 2, box[0][0] + box[1][0] / 2, box[0][1] + box[1][1] / 2, center[0], center[1], timestamp, None, None, object_ids[i], event_inference_time, kalman_inference_time, total_inference_time, new_center_kalman[0], new_center_kalman[1]]
        tracking_end_time = time.perf_counter()
        tracking_time = tracking_end_time - tracking_start_time
        single_tracking_times.append(tracking_time)
        print(f"Tracked box {i + 1} center: {center}, tracking time: {tracking_time:.6f} seconds")

    end_time = time.perf_counter()
    frame_count += 1
    if frame_count % 30 == 0:  # Calculate FPS every 30 frames
        current_time = time.perf_counter()
        fps = frame_count / (current_time - fps_start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        fps_start_time = current_time

    # Capture frame data from DAVIS346
    frame = capture.getNextFrame()
    if frame is not None:
        out.write(frame.image)

# Register callback every 1 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=10), slicing_callback)

# Main function
def main():
    global glip_labels, start_tracking_time, tracking_duration, is_tracking, df, out, capture, single_tracking_times, object_ids, kalman_filters

    # Manually input description text
    caption_zh = input("Please enter the description text: ")
    english_caption = translate_zh_to_en(caption_zh)
    print(f"Translation result: {english_caption}")

    previous_time = time.time()

    # Capture the first frame using the event camera
    capture = dv.io.CameraCapture()
    first_frame_captured = False

    start_time_capture_first_frame = time.time()

    # Run the loop while the event camera is still connected
    while capture.isRunning() and not first_frame_captured:
        # Read a frame from the event camera
        frame = capture.getNextFrame()

        # the latest available frame or if no data is available, returns `None`
        if frame is not None:
            # Save the first frame as an image file
            image_path = "first_frame.png"
            cv.imwrite(image_path, frame.image)

            # Set the flag to True to stop the loop
            first_frame_captured = True

    end_time_capture_first_frame = time.time()
    print(f"Time to capture first frame: {end_time_capture_first_frame - start_time_capture_first_frame:.2f} seconds")

    # Initialize video writer for color video
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('color_video.avi', fourcc, 30.0, (346, 260))  # Adjust frame size as needed

    # Load the captured image and perform GLIP inference
    if first_frame_captured:
        start_time_glip_inference = time.time()

        color_image = cv.imread(image_path)
        color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)

        current_time = time.time()
        inference_interval = current_time - previous_time
        previous_time = current_time

        inference_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Record inference time

        boxes, scores, labels_names, colors = glip_inference(color_image, english_caption)
        boxes, scores, labels_names = filter_highest_confidence_boxes(boxes, scores, labels_names)

        end_time_glip_inference = time.time()
        print(f"GLIP inference time: {end_time_glip_inference - start_time_glip_inference:.2f} seconds")

        # Update initial label boxes for tracking
        initial_label_boxes = np.array([[[((x1 + x2) / 2), ((y1 + y2) / 2)], [(x2 - x1), (y2 - y1)]] for x1, y1, x2, y2 in boxes], dtype=np.float32)
        global tracked_boxes
        tracked_boxes = initial_label_boxes.copy()

        glip_labels = labels_names

        # Assign unique IDs to detected objects
        object_ids = list(range(1, len(labels_names) + 1))

        # Initialize Kalman filters for each box
        kalman_filters = {obj_id: KalmanFilter(tracked_boxes[i][0], np.array([0, 0])) for i, obj_id in enumerate(object_ids)}

        # Collect initial GLIP data
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            timestamp = int(datetime.now().timestamp() * 1e6)
            df.loc[len(df)] = ['GLIP', labels_names[i], scores[i], x1, y1, x2, y2, center_x, center_y, timestamp, inference_time, inference_interval, object_ids[i], None, None, None, None, None]
            print(f"Box {i + 1}:")
            print(f"  - ID: {object_ids[i]}")
            print(f"  - Labels: {labels_names[i]}")
            print(f"  - Score: {scores[i]:.2f}")
            print(f"  - Corners: (x1: {x1}, y1: {y1}), (x2: {x2}, y2: {y2})")
            print(f"  - Center: (x: {center_x}, y: {center_y})")
            print(f"  - Timestamp: {timestamp}")
            print(f"  - Inference Time: {inference_time}")
            print(f"  - Inference Interval: {inference_interval:.6f} seconds")

        # Start event tracking after displaying GLIP results
        start_tracking_time = time.perf_counter()
        tracking_duration = 10  # Set tracking duration to 10 seconds
        is_tracking = True
        while capture.isRunning() and is_tracking:
            events = capture.getNextEventBatch()
            if events is not None:
                slicer.accept(events)

    end_time_tracking = time.perf_counter()
    print(f"Total tracking time: {end_time_tracking - start_tracking_time:.2f} seconds")
    if single_tracking_times:
        average_tracking_time = sum(single_tracking_times) / len(single_tracking_times)
        print(f"Average single tracking box time: {average_tracking_time:.6f} seconds")

    # Save the dataframe to an xlsx file
    start_time_save_xlsx = time.time()
    df.to_excel('E:/event/detected_objects_event_noimg.xlsx', index=False)
    end_time_save_xlsx = time.time()
    print(f"Time to save xlsx: {end_time_save_xlsx - start_time_save_xlsx:.2f} seconds")
    print("Tracking data saved to detected_objects_event_noimg.xlsx")

    # Release resources
    cv.destroyAllWindows()

    # Draw bounding boxes on saved video
    start_time_draw_boxes = time.time()
    draw_boxes_on_saved_video('color_video.avi', 'E:/event/detected_objects_event_noimg.xlsx', 'color_video_with_boxes.avi')
    end_time_draw_boxes = time.time()
    print(f"Time to draw boxes on video: {end_time_draw_boxes:.2f} seconds")

def draw_boxes_on_saved_video(input_video_path, excel_path, output_video_path):
    # Open the input video
    cap = cv.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Initialize video writer for output video
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Read the bounding box data from the Excel file
    df = pd.read_excel(excel_path)
    box_data = df[['X1', 'Y1', 'X2', 'Y2', 'Label', 'Score', 'ID']].values
    total_boxes = len(box_data)

    # Distribute boxes evenly across frames
    boxes_per_frame = max(1, total_boxes // frame_count)

    frame_index = 0
    box_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw boxes on the current frame
        if box_index < total_boxes:
            for _ in range(boxes_per_frame):
                if box_index < total_boxes:
                    x1, y1, x2, y2, label, score, obj_id = box_data[box_index]
                    color = (0, 255, 0)  # Green color for bounding boxes
                    frame = draw_boxes_on_frame(frame, [(x1, y1, x2, y2)], [label], [score], [color], [obj_id])
                    box_index += 1

        # Write the frame with bounding boxes to the output video
        out.write(frame)
        frame_index += 1

    # Release video resources
    cap.release()
    out.release()
    print("Finished drawing boxes on video")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
