import warnings
import os
import cv2
import numpy as np
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from transformers import logging
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

class Colors:
    def __init__(self):
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

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
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

def glip_inference(image_, caption_):
    colors_ = Colors()
    preds = glip_demo.compute_prediction(image_, caption_)
    top_preds = glip_demo._post_process(preds, threshold=0.6)
    labels = top_preds.get_field("labels").tolist()
    scores = top_preds.get_field("scores").tolist()
    boxes = top_preds.bbox.detach().cpu().numpy()
    colors = [colors_(idx) for idx in labels]
    labels_names = glip_demo.get_label_names(labels)
    ids = list(range(len(boxes)))  # Assign unique IDs to each detected object
    return boxes, colors, ids

class KalmanFilter:
    def __init__(self, initial_state, max_shift=4):
        self.kf = cv2.KalmanFilter(6, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
                                             [0, 1, 0, 1, 0, 0.5],
                                             [0, 0, 1, 0, 1, 0],
                                             [0, 0, 0, 1, 0, 1],
                                             [0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.statePost = np.array([[initial_state[0]], [initial_state[1]], [0], [0], [0], [0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.max_shift = max_shift

    def predict(self):
        prediction = self.kf.predict()[:2].reshape(-1)
        if np.any(np.isnan(prediction)):
            prediction = self.kf.statePost[:2].reshape(-1)
        return prediction

    def correct(self, coords):
        coords = np.array([[coords[0]], [coords[1]]], dtype=np.float32)
        self.kf.correct(coords)

    def adjust_noise_covariance(self, measurement, prediction):
        error = np.linalg.norm(measurement - prediction)
        if error > 3.0:
            self.kf.processNoiseCov *= 1.1
            self.kf.measurementNoiseCov *= 1.1
        else:
            self.kf.processNoiseCov *= 0.9
            self.kf.measurementNoiseCov *= 0.9

    def limit_shift(self, new_center, old_center):
        shift_x = np.clip(new_center[0] - old_center[0], -self.max_shift, self.max_shift)
        shift_y = np.clip(new_center[1] - old_center[1], -self.max_shift, self.max_shift)
        return old_center[0] + shift_x, old_center[1] + shift_y

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def filter_boxes_by_average_value(boxes, npy_file, threshold):
    matrix = np.load(npy_file)
    absolute_matrix = np.abs(matrix)
    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        sub_matrix = absolute_matrix[y1:y2, x1:x2]
        average_value = np.mean(sub_matrix)
        print(f"Box: {box}, Average Value: {average_value}")  # Debug information
        if average_value >= threshold:
            filtered_boxes.append(tuple(box))  # Convert to tuple for comparison
    return filtered_boxes

def process_and_track_npy_files(npy_dir, start_id, end_id, boxes, colors, ids, output_video_path, csv_path, glip_inference_time):
    images = []
    total_files = (end_id - start_id + 1) * 25
    file_count = 0

    kalman_filters = {obj_id: KalmanFilter(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)) for obj_id, box in zip(ids, boxes)}
    box_sizes = {obj_id: (box[2] - box[0], box[3] - box[1]) for obj_id, box in zip(ids, boxes)}
    previous_distances = {}
    center_points = {obj_id: [] for obj_id in ids}

    start_time = time.perf_counter()
    tracking_time_start = time.perf_counter()
    frame_times = []

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['GLIP Inference Time (s)'])
        csv_writer.writerow([glip_inference_time])
        csv_writer.writerow(['Frame Number', 'Time Taken (s)', 'Object ID', 'Center X', 'Center Y', 'Predicted Center X', 'Predicted Center Y', 'Loss', 'Box Inference Time (s)'])

        for batch in range(start_id, end_id + 1):
            for i in range(25):
                npy_file = f'COP_{batch}_TD_{i}.npy'
                file_path = os.path.join(npy_dir, npy_file)
                if not os.path.exists(file_path):
                    continue

                matrix = np.load(file_path)
                frame_start_time = time.perf_counter()
                absolute_matrix = np.abs(matrix)
                normalized_absolute = np.clip(absolute_matrix / absolute_matrix.max(), 0, 1)

                new_boxes = []
                new_ids = []

                for obj_id in ids:
                    box_start_time = time.perf_counter()

                    kf = kalman_filters[obj_id]
                    x1, y1, x2, y2 = map(int, boxes[obj_id])
                    box_width, box_height = box_sizes[obj_id]

                    max_sum = -np.inf
                    max_pos = (int(x1), int(y1))
                    for y in range(max(int(y1 - 3), 0), min(int(y1 + 3), absolute_matrix.shape[0] - int(box_height)) + 1):
                        for x in range(max(int(x1 - 3), 0), min(int(x1 + 3), absolute_matrix.shape[1] - int(box_width)) + 1):
                            sub_matrix = absolute_matrix[y:y + int(box_height), x:x + int(box_width)]
                            sub_sum = np.sum(sub_matrix)
                            if sub_sum > max_sum:
                                max_sum = sub_sum
                                max_pos = (x, y)

                    max_box = (max_pos[0], max_pos[1], max_pos[0] + int(box_width), max_pos[1] + int(box_height))
                    new_boxes.append(max_box)
                    new_ids.append(obj_id)

                    center_measurement = ((max_box[0] + max_box[2]) / 2, (max_box[1] + max_box[3]) / 2)
                    predicted_center = kf.predict()
                    kf.correct(center_measurement)
                    kf.adjust_noise_covariance(center_measurement, predicted_center)

                    if np.any(np.isnan(predicted_center)):
                        print(f"Warning: NaN detected in predicted_center for object ID {obj_id}")
                        predicted_center = center_measurement

                    limited_center = kf.limit_shift(predicted_center, ((boxes[obj_id][0] + boxes[obj_id][2]) / 2, (boxes[obj_id][1] + boxes[obj_id][3]) / 2))

                    use_kalman = False
                    for other_id, other_center in previous_distances.items():
                        if other_id != obj_id:
                            distance = calculate_distance(predicted_center, other_center)
                            if distance < 40:
                                use_kalman = True
                                previous_distances[(obj_id, other_id)] = distance
                            elif (obj_id, other_id) in previous_distances and distance > 40:
                                use_kalman = False

                    if use_kalman:
                        predicted_box = (
                            int(limited_center[0] - box_width / 2),
                            int(limited_center[1] - box_height / 2),
                            int(limited_center[0] + box_width / 2),
                            int(limited_center[1] + box_height / 2)
                        )
                        boxes[obj_id] = predicted_box
                    else:
                        boxes[obj_id] = max_box

                    center_points[obj_id].append(limited_center)
                    box_end_time = time.perf_counter()
                    frame_end_time = time.perf_counter()
                    # Calculate and record the loss
                    loss = calculate_distance(center_measurement, predicted_center)
                    print(f'ID: {obj_id}, Frame: {file_count + 1}, Loss: {loss:.2f}')

                    # Record data for this object
                    box_inference_time = box_end_time - box_start_time
                    frame_time = frame_end_time - frame_start_time
                    csv_writer.writerow([file_count + 1, frame_time, obj_id, center_measurement[0], center_measurement[1], predicted_center[0], predicted_center[1], loss, box_inference_time])

                frame = cv2.normalize(normalized_absolute, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                for max_box, obj_id in zip(new_boxes, new_ids):
                    cv2.rectangle(frame, (max_box[0], max_box[1]), (max_box[2], max_box[3]), (255, 0, 0), 2)
                    cv2.putText(frame, f'ID: {obj_id}', (max_box[0], max_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                images.append(frame)
                file_count += 1

                if len(frame_times) > 100:
                    frame_times.pop(0)
                if sum(frame_times) > 0:
                    real_time_fps = len(frame_times) / sum(frame_times)
                    progress = (file_count / total_files) * 100
                    print(f'Processing file {file_count}/{total_files} ({progress:.2f}%) - Real-time FPS: {real_time_fps:.2f}')
    
    tracking_time_end = time.perf_counter()
    tracking_total_time = tracking_time_end - tracking_time_start

    height, width = images[0].shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 750, (width, height))
    for image in images:
        out.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    out.release()

    final_fps = total_files / tracking_total_time
    print(f'Final average FPS (tracking only): {final_fps:.2f}')

def process_directory(npy_dir, start_id, end_id, caption, output_video_path, csv_path, threshold=0.02):#0.02
    bmp_file = os.path.join(npy_dir, f'COP_{start_id}.bmp')
    image = cv2.imread(bmp_file)
    if image is None:
        print(f"Error: Unable to read image file {bmp_file}")
        return

    start_glip_time = time.perf_counter()
    boxes, colors, ids = glip_inference(image, caption)
    end_glip_time = time.perf_counter()
    glip_inference_time = end_glip_time - start_glip_time
    print(f'Time taken for GLIP inference: {glip_inference_time:.2f} seconds')

    # Filter boxes based on average value in the first TD file
    first_npy_file = os.path.join(npy_dir, f'COP_{start_id}_TD_0.npy')
    filtered_boxes = filter_boxes_by_average_value(boxes, first_npy_file, threshold)
    if not filtered_boxes:
        print("No boxes passed the average value threshold.")
        return

    # Update boxes and ids to only include filtered ones
    filtered_ids = [i for i, box in enumerate(boxes) if tuple(box) in filtered_boxes]
    filtered_colors = [colors[i] for i in filtered_ids]
    filtered_boxes = [boxes[i] for i in filtered_ids]
    filtered_ids = list(range(len(filtered_boxes)))  # Re-assign IDs to filtered boxes

    process_and_track_npy_files(npy_dir, start_id, end_id, filtered_boxes, filtered_colors, filtered_ids, output_video_path, csv_path, glip_inference_time)

def calculate_3d_points(file1, file2, fx1, fx2, baseline):
    data1 = pd.read_csv(file1, skiprows=2)
    data2 = pd.read_csv(file2, skiprows=2)

    data1.sort_values(by='Frame Number', inplace=True)
    data2.sort_values(by='Frame Number', inplace=True)

    points_3d = []

    # Computing depth
    for idx, row in data1.iterrows():
        frame_number = row['Frame Number']
        center_x1 = row['Center X']
        center_y1 = row['Center Y']
        
        # Match Frame Number
        match = data2[data2['Frame Number'] == frame_number]
        
        if not match.empty:
            center_x2 = match.iloc[0]['Center X']
            center_y2 = match.iloc[0]['Center Y']
            
            # Disparity
            disparity = center_x1 - center_x2
            
            # Depth
            fx_avg = (fx1 + fx2) / 2
            depth = fx_avg * baseline / disparity if disparity != 0 else float('inf')
            
            if depth != float('inf'):
                x = center_x1
                y = center_y1
                z = depth
                points_3d.append([x, y, z])
            
            print(f"Frame {frame_number} - Depth: {depth}")

    # from 3D points to NumPy array
    points_3d = np.array(points_3d)

    # Save
    if points_3d.size > 0:
        df_points_3d = pd.DataFrame(points_3d, columns=['X', 'Y', 'Z'])
        output_file = '3d_points.csv'
        df_points_3d.to_csv(output_file, index=False)
        print(f"3D points saved to {output_file}")

        X = points_3d[:, 0]
        Y = points_3d[:, 1]
        Z = points_3d[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Z, Y, label='3D Points')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Z axis')
        ax.set_zlabel('Y axis')
        ax.set_xlim(0, 640)  # X axis range
        ax.set_ylim(0, 1000)  # Z axis range
        ax.set_zlim(0, 320)  # Y axis range
        plt.legend()
        plt.show()
    else:
        print("No valid 3D points to plot.")

def main():

    caption = 'there is a ball' # text capture

    directories = [
        ('path/to/database', 79, 81, 'output_video1.mp4', 'tracking_data1.csv'),
        ('path/to/database', 79, 81, 'output_video2.mp4', 'tracking_data2.csv')
    ]#79 and 81 here are the frame numbers of images

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_directory, npy_dir, start_id, end_id, caption, output_video_path, csv_path, 0.0) 
                   for npy_dir, start_id, end_id, output_video_path, csv_path in directories]
        for future in futures:
            future.result()

    # Calculate 3D points
    calculate_3d_points('tracking_data2.csv', 'tracking_data1.csv', 993, 495, 130)

if __name__ == '__main__':
    main()
