import argparse
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def parse_arguments():
    parser = argparse.ArgumentParser(description='Need to enter video file path')
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to video file')

    args = parser.parse_args()
    return args.video


def create_detector():
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    model_path = 'pose_landmarker_heavy.task'
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )
    pose_detector = vision.PoseLandmarker.create_from_options(options)
    return pose_detector


def annotate_image_with_landmarks(rgb_image, pose_detections):
    pose_landmarks_list = pose_detections.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def process_video_with_landmarks(path):
    if path:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error opening video file.")
            exit(-1)

        video_file_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_timestamp_ms = int(1000 * frame_count / video_file_fps)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            pose_landmarker_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            annotated_image = annotate_image_with_landmarks(frame, pose_landmarker_result)
            cv2.imshow('annotated_image', annotated_image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    detector = create_detector()
    video_path = parse_arguments()
    process_video_with_landmarks(video_path)

