import argparse
import cv2
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime


class VideoGUI:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self.frame = None
        self.cursor_positions = []
        self.saved_convex_shape = []
        self.shape_is_saved = False
        self.mode = AppMode.PLAY
        self.frame_copy = None

    def init_video_gui(self):
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print("Error opening video file.")
            exit(-1)

        cv2.namedWindow("GUI")
        cv2.setMouseCallback("GUI", self.mouse_callback)

        self.frame_count = 0
        self.frame = None
        video_file_fps = self.cap.get(cv2.CAP_PROP_FPS)
        return video_file_fps

    def terminate_video_gui(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == AppMode.REGION_OF_INTEREST:
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.cursor_positions) == 10:
                    self.cursor_positions.pop(0)
                self.cursor_positions.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and self.cursor_positions:
                self.cursor_positions.pop()

    def set_mode(self, mode):
        self.mode = mode

    def key_handler(self):
        key = cv2.waitKey(1) & 0xFF

        if key == 32:
            if self.mode == AppMode.REGION_OF_INTEREST:
                self.set_mode(AppMode.PLAY)
            else:
                self.set_mode(AppMode.REGION_OF_INTEREST)
                print("Region of interest mode enabled!")
        elif key == 13:
            if self.mode == AppMode.DETECTION:
                self.set_mode(AppMode.PLAY)
            elif is_convex(self.cursor_positions):
                print("Convex shape saved successfully!")
                self.saved_convex_shape = self.cursor_positions
                self.shape_is_saved = True
                self.set_mode(AppMode.DETECTION)
                print("Detection mode enabled!")
        elif key == 27:
            print("User chose to exit.")
            return "exit"

    def draw_connections(self, current_frame):
        if len(self.cursor_positions) > 1:
            for i in range(len(self.cursor_positions) - 1):
                cv2.line(current_frame, self.cursor_positions[i], self.cursor_positions[i + 1], (212, 255, 127), 2)
            if len(self.cursor_positions) > 2:
                cv2.line(current_frame, self.cursor_positions[-1], self.cursor_positions[0], (212, 255, 127), 2)

    def draw_roi(self, current_frame):
        self.frame_copy = current_frame.copy()
        draw_positions(self.frame_copy, self.cursor_positions)
        self.draw_connections(self.frame_copy)
        if not is_convex(self.cursor_positions):
            draw_text_on_frame(self.frame_copy, "Shape must be convex to save!", 300, 160)
        return self.frame_copy

    def roi_mode(self):
        self.frame_copy = self.draw_roi(self.frame)
        cv2.imshow("GUI", self.frame_copy)

    def play_mode(self):
        cv2.imshow("GUI", self.frame)
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error reading video frame.")
            exit(-1)
        self.frame_count += 1


class AppMode(Enum):
    PLAY = 0
    REGION_OF_INTEREST = 1
    DETECTION = 2


def draw_positions(current_frame, positions):
    for pos in positions:
        cv2.circle(current_frame, pos, 5, (255, 0, 0), -1)


def draw_text_on_frame(current_frame, text, x, y):
    cv2.putText(
        current_frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        2,
    )


def is_convex(positions):
    if len(positions) > 2:
        return cv2.isContourConvex(np.array(positions))
    else:
        return False


def parse_argument():
    parser = argparse.ArgumentParser(description="Need to enter video file path")
    parser.add_argument("-v", "--video", type=str, required=True, help="Path to video file")

    args = parser.parse_args()
    return args.video


def text_on_image(fps_value, inf_time, img, plane_detect):
    text_fps = f"FPS: {fps_value}"
    text_inf = f"inf.time: {inf_time} ms"
    if plane_detect.intersection_status:
        text_status = f"status: crossed"
    else:
        text_status = f"status: not"
    color = (200, 112, 219)
    cv2.putText(
        img,
        text_fps,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    cv2.putText(
        img,
        text_inf,
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    cv2.putText(
        img,
        text_status,
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def draw_skeleton(gui, model_detector, plane_detect):
    coord = model_detector.to_find_driver_px_coord(gui.frame, plane_detect)
    if coord:
        draw_positions(gui.frame, coord)
        for start, end in model_detector.POSE_CONNECTIONS:
            if start < len(coord) and end < len(coord):
                cv2.line(gui.frame, coord[start], coord[end], (0, 0, 255), 2)


def display(gui, video_file_fps, fps, model_detector, detector, plane_detect):
    infer_time = model_detector.determine_skeleton_and_inference_time(
        gui.frame, gui.frame_count, video_file_fps, detector, plane_detect
    )
    draw_skeleton(gui, model_detector, plane_detect)
    text_on_image(fps, infer_time, gui.frame, plane_detect)


def det_mode(gui, plane_detect, fig):
    draw_positions(gui.frame, gui.cursor_positions)
    gui.draw_connections(gui.frame)
    frame_height, frame_width, _ = gui.frame.shape
    intersection_result = plane_detect.result_of_finding_intersection(gui.cursor_positions, frame_height, frame_width)
    if intersection_result is not None:
        plt_img = intersection_result.plot_chart(fig)
        h, w, _ = plt_img.shape
        y_offset, x_offset = frame_height - h, 0
        gui.frame[y_offset : y_offset + h, x_offset : x_offset + w] = plt_img

    return gui.frame


def run_video_gui(gui, model_detector, plane_detect):
    fig = plt.figure(figsize=[3, 3])
    video_file_fps = gui.init_video_gui()
    timer = datetime.now()

    detector = model_detector.create_detector()

    ret, gui.frame = gui.cap.read()
    while gui.key_handler() != "exit":
        plane_detect.intersection_status = False
        delta = datetime.now() - timer
        fps = int(1.0 / max(0.0001, delta.total_seconds()))
        timer = datetime.now()

        if gui.mode == AppMode.REGION_OF_INTEREST:
            gui.roi_mode()
        elif gui.shape_is_saved and gui.mode == AppMode.DETECTION:
            gui.frame = det_mode(gui, plane_detect, fig)
            display(gui, video_file_fps, fps, model_detector, detector, plane_detect)
            gui.play_mode()
        else:
            display(gui, video_file_fps, fps, model_detector, detector, plane_detect)
            gui.play_mode()

    gui.terminate_video_gui()

