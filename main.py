# Импортируем необходимые модули.
import argparse
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Добавляем поддержку аргументов командной строки.
def parse_arguments():
    parser = argparse.ArgumentParser(description='Need to enter video file path')
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to video file')

    args = parser.parse_args()
    return args.video


def annotate_image_with_landmarks(rgb_image, pose_detections):
    pose_landmarks_list = pose_detections.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Обходим обнаруженные позы для визуализации.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Отображаем ориентиры позы.
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


if __name__ == '__main__':
    model_path = 'pose_landmarker_heavy.task'
    BaseOptions = mp.tasks.BaseOptions
    ImageClassifier = mp.tasks.vision.ImageClassifier
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Создаем объект PoseLandmarker.
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )
    pose_detector = vision.PoseLandmarker.create_from_options(options)
    video_path = parse_arguments()
    # Если указан путь к видеофайлу, то читаем его.
    if video_path:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error opening video file.")
            exit(-1)

        video_file_fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0  # initialize a frame counter

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # calculate frame timestamp in milliseconds
            frame_timestamp_ms = int(1000 * frame_count / video_file_fps)

            # cv2.imshow('Frame', frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            pose_landmarker_result = pose_detector.detect_for_video(mp_image, frame_timestamp_ms)
            annotated_image = annotate_image_with_landmarks(frame, pose_landmarker_result)
            cv2.imshow('annotated_image', annotated_image)

            if cv2.waitKey(1) & 0xFF == 27:  # Изменение условия сравнения
                break

            frame_count += 1  # increment the frame counter

        cap.release()
        cv2.destroyAllWindows()
