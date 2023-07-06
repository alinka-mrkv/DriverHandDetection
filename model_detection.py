import mediapipe as mp
from mediapipe.tasks.python import vision
import time
from mediapipe import solutions


class ModelDetection:
    POSE_CONNECTIONS = frozenset(solutions.pose.POSE_CONNECTIONS)

    def __init__(self):
        pass

    def create_detector(self):
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        model_path = "pose_landmarker_lite.task"
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
        )
        pose_detector = vision.PoseLandmarker.create_from_options(options)
        return pose_detector

    def determine_skeleton_and_inference_time(self, frame, frame_count, video_file_fps, detector, plane_detect):
        frame_timestamp_ms = int(1000 * frame_count / video_file_fps)
        start_time = time.time_ns()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        pose_landmarker_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        inference_time = (time.time_ns() - start_time) / 1000000

        new_world_landmarks = pose_landmarker_result.pose_world_landmarks
        new_landmarks = pose_landmarker_result.pose_landmarks
        if new_world_landmarks:
            plane_detect.world_landmarks = new_world_landmarks
        if new_landmarks:
            plane_detect.normalized_landmarks = new_landmarks
        return inference_time

    def to_find_driver_px_coord(self, frame, plane_detect):
        normalized_landmarks = plane_detect.normalized_landmarks
        world_landmarks = plane_detect.world_landmarks

        height, width, _ = frame.shape
        if not normalized_landmarks:
            return 0
        driver_px_coordinates = []
        world_landmarks_copy = []
        worldlandmark = world_landmarks[0]
        for landmark in normalized_landmarks:
            for i, result in enumerate(landmark):
                point_px = solutions.drawing_utils._normalized_to_pixel_coordinates(result.x, result.y, width, height)

                if point_px:
                    driver_px_coordinates.append(list(point_px))
                    world_landmarks_copy.append(worldlandmark[i])

        plane_detect.world_landmarks_copy = world_landmarks_copy
        plane_detect.driver_px_coordinates = driver_px_coordinates
        return driver_px_coordinates
