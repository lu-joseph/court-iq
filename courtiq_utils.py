import cv2
import supervision as sv

import numpy as np

from typing import Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
from ultralytics.engine.results import Probs, Keypoints, OBB, Boxes, Masks
import matplotlib.path as mpltPath
import inference
import os
import csv
import matplotlib.pyplot as plt

left_foot_kid = 15
right_foot_kid = 16
pose_model = YOLO("yolo11x-pose.pt") 
ROBOFLOW_API_KEY=os.getenv('ROBOFLOW_API_KEY')
court_detection_model = inference.get_model("badminton-court-detection-cfgah/3",api_key=ROBOFLOW_API_KEY)


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)
    
    
@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int

    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r

def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
    return image

def draw_rects(frame, rects, color=Color(r=0,g=255,b=0)):
    annotated_frame = frame
    for rect in rects:
        annotated_frame = draw_rect(image=annotated_frame,
                                    rect=rect,
                                    color=color,
                                    thickness=2)
    return annotated_frame

def group_and_average(arr, threshold):
    groups = []  # Store grouped numbers

    current_group = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] <= threshold:
            current_group.append(arr[i])
        else:
            groups.append(current_group)  # Save the completed group
            current_group = [arr[i]]  # Start a new group

    groups.append(current_group)  # Add the last group

    # Calculate the average for each group
    averages = [int(np.mean(group)) for group in groups]
    return averages

def detections_from_yolov11(results) -> sv.Detections:
    return sv.Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int),
        tracker_id=results.boxes.id.int().cpu().numpy() if results.boxes.id is not None else None)

def get_points_from_court_prediction(class_name: str, predictions):
    side_prediction = [p for p in predictions if p.class_name == class_name]
    if len(side_prediction) != 1: return None
    return side_prediction[0].points

# 
def filter_results(results, i, j):
    results.masks = Masks(results.masks.data[i:j], results.orig_shape) if results.masks else None
    results.boxes = Boxes(results.boxes.data[i:j], results.orig_shape) if results.boxes else None
    results.probs = Probs(results.probs.data[i:j], results.orig_shape) if results.probs else None
    results.keypoints = Keypoints(results.keypoints.data[i:j], results.orig_shape) if results.keypoints else None
    results.obb = OBB(results.obb.data[i:j], results.orig_shape) if results.obb else None

def get_court_path(frame, side: str):
    court_results = court_detection_model.infer(frame)[0]
    near_side_points = get_points_from_court_prediction(side, court_results.predictions)
    polygon = [[point.x, point.y] for point in near_side_points]
    path = mpltPath.Path(polygon)
    return path

# returns 
def get_players_in_path(pose_results, path: mpltPath.Path, limit_one: bool):
    key_points_list = pose_results.keypoints.data.cpu().numpy()
    num_ppl = len(key_points_list)
    players = []
    for i in range(num_ppl):
        key_points = key_points_list[i]
        left_foot = key_points[left_foot_kid]
        right_foot = key_points[right_foot_kid]
        if path.contains_points([[foot[0], foot[1]] for foot in [left_foot, right_foot]]).any():
            players.append(i)
            if limit_one: 
                break
    return players

def get_pose_results_for_path(frame, path):
    pose_results = pose_model(frame, verbose=False)[0]
    players_in_path = get_players_in_path(pose_results, path, limit_one=True)
    if not players_in_path: return None
    player_idx = players_in_path[0]
    filter_results(pose_results, player_idx, player_idx + 1)
    return pose_results

def get_frame_xys_from_csv(file):
    frame_to_xy = {}
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frame_to_xy[int(row["Frame"])] = [int(row["X"]), int(row["Y"])]
    return frame_to_xy

def get_shot_frames_from_xys(frame_to_xy):
    frame_indices = np.arange(len(frame_to_xy))
    y = [f[1] for f in frame_to_xy.values()]
    y_1 = np.gradient(y)
    y_2 = np.gradient(y_1)
    shot_frames = group_and_average([idx for idx in frame_indices if y_2[idx] < -3],threshold=5)
    return shot_frames

def get_xys_for_shot_frame(shot_frame_idx):
    """
    Returns: [right_elbow, right_wrist]
    """
    right_shoulder_kid = 6
    right_elbow_kid = 8
    right_wrist_kid = 10

    # all relative to right_shoulder
    right_elbow = []
    right_wrist = []
    for frame in sv.get_video_frames_generator(source_path="rally.mp4",start=shot_frame_idx-10,end=shot_frame_idx+10):
        path = get_court_path(frame, 'near-side')
        pose_results = get_pose_results_for_path(frame, path)
        if not pose_results:
            right_elbow.append(right_elbow[-1] if right_elbow else [0,0])
            right_wrist.append(right_wrist[-1] if right_wrist else [0,0])
            continue
        skeleton = pose_results.keypoints.xy.cpu().numpy()[0]
        right_shoulder = skeleton[right_shoulder_kid]
        right_elbow.append(skeleton[right_elbow_kid] - right_shoulder)
        right_wrist.append(skeleton[right_wrist_kid] - right_shoulder)

    return [right_elbow, right_wrist]

def get_bird_xys_for_shot_frame(shot_frame_idx, frame_to_xy):
    xys = []
    for i in range(shot_frame_idx - 10, shot_frame_idx + 10):
        xys.append(frame_to_xy[i])
    return xys


def plot_limb(limb_xys, shot_frame_idx, title="", save_dir=None, file_name=None):
    """
    limb_xys: list[x,y] 
    """
    x = np.arange(shot_frame_idx - 10, shot_frame_idx + 10)
    plt.plot(x, [f[0] for f in limb_xys],color="red",label="x")
    plt.plot(x, [f[1] for f in limb_xys],color="green",label="y")
    plt.title(title)
    plt.legend()
    plt.show()
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + '/' +  file_name)


def split_into_shots(video_file, frames_before_shot=10,frames_after_shot=10):
    """
    ASSUMES CSV HAS BEEN CREATED
    """
    video_name = video_file.split('/')[-1][:-4]
    print(f'video name: {video_name}')
    video_info = sv.VideoInfo.from_video_path(video_path=video_file)
    csv_path = f'TrackNetV3/pred_result/{video_name}_ball.csv'
    if not os.path.exists(csv_path):
        print("csv not found")
        return
    frame_xys = get_frame_xys_from_csv(csv_path)
    shot_frames = get_shot_frames_from_xys(frame_xys)

    for idx,shot_idx in enumerate(shot_frames):
        start_frame = max(shot_idx-frames_before_shot,0)
        end_frame = min(shot_idx+frames_after_shot,video_info.total_frames - 1)
        total_frames = end_frame - start_frame + 1
        shot_video_info = sv.VideoInfo(width=video_info.width,height=video_info.height,fps=video_info.fps,total_frames=total_frames)
        output_directory = f'./output/videos/{video_name}'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_path = f'{output_directory}/shot{idx}.mp4'
        
        with sv.VideoSink(target_path=output_path,video_info=shot_video_info, codec="mp4v") as sink:
            for frame in sv.get_video_frames_generator(source_path=video_file,start=start_frame,end=end_frame):
                sink.write_frame(frame)
        print(f'done shot {idx} at {output_path}',end='\r')
    print('\ndone')

def plot_shot(shot_idx, csv_file, type_of_shot, video_num, save_dir=None):
    frame_xys = get_frame_xys_from_csv(csv_file)
    shot_frames = get_shot_frames_from_xys(frame_xys)
    shot_frame_idx = shot_frames[shot_idx]
    print(f'shot happens at frame {shot_frame_idx}')
    right_elbow, right_wrist = get_xys_for_shot_frame(shot_frame_idx)
    bird = get_bird_xys_for_shot_frame(shot_frame_idx, frame_xys)

    plot_limb(right_elbow, shot_frame_idx, f'elbow during {type_of_shot} (shot {shot_idx})',save_dir + f'/elbow', f'{video_num}_{shot_idx}.png')
    plot_limb(right_wrist, shot_frame_idx, f'wrist during {type_of_shot} (shot {shot_idx})',save_dir + f'/wrist', f'{video_num}_{shot_idx}.png')
    plot_limb(bird, shot_frame_idx, f'bird during {type_of_shot}, (shot {shot_idx})',save_dir + f'/bird', f'{video_num}_{shot_idx}.png')