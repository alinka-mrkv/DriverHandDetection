import argparse
import cv2
import numpy as np
from enum import Enum


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
                self.mode = AppMode.PLAY
            else:
                self.set_mode(AppMode.REGION_OF_INTEREST)
                print("Region of interest mode enabled!")
        elif key == 13:
            if is_convex(self.cursor_positions):
                if self.mode == AppMode.REGION_OF_INTEREST:
                    print("Convex shape saved successfully!")
                    self.saved_convex_shape = self.cursor_positions
                    self.shape_is_saved = True
                    self.set_mode(AppMode.DETECTION)
                    print("Detection mode enabled!")
        elif key == 27:
            print("User chose to exit.")
            return "exit"

    def draw_positions(self, current_frame):
        for pos in self.cursor_positions:
            cv2.circle(current_frame, pos, 5, (255, 0, 0), -1)

    def draw_connections(self, current_frame):
        if len(self.cursor_positions) > 1:
            for i in range(len(self.cursor_positions) - 1):
                cv2.line(current_frame, self.cursor_positions[i], self.cursor_positions[i + 1], (212, 255, 127), 2)
            if len(self.cursor_positions) > 2:
                cv2.line(current_frame, self.cursor_positions[-1], self.cursor_positions[0], (212, 255, 127), 2)

    def draw_roi(self, current_frame):
        self.frame_copy = current_frame.copy()
        self.draw_positions(self.frame_copy)
        self.draw_connections(self.frame_copy)
        if not is_convex(self.cursor_positions):
            draw_text_on_frame(self.frame_copy, "Shape must be convex to save!", 300, 160)
        return self.frame_copy

    def roi_mode(self):
        self.frame_copy = self.draw_roi(self.frame)
        cv2.imshow("GUI", self.frame_copy)

    def play_mode(self):
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error reading video frame.")
            exit(-1)
        cv2.imshow("GUI", self.frame)
        self.frame_count += 1


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


def det_mode():
    image = np.empty((720, 1280, 3), dtype=np.uint8)
    draw_text_on_frame(image, "detection mode not implemented!", 260, 120)
    cv2.imshow("GUI", image)


def run_video_gui(gui):
    gui.init_video_gui()
    while gui.key_handler() != "exit":
        if gui.mode == AppMode.REGION_OF_INTEREST:
            gui.roi_mode()
        elif gui.mode == AppMode.DETECTION:
            det_mode()
        else:
            gui.play_mode()

    gui.terminate_video_gui()


class AppMode(Enum):
    PLAY = 0
    REGION_OF_INTEREST = 1
    DETECTION = 2


if __name__ == "__main__":
    video_path = parse_argument()
    gui = VideoGUI(video_path)
    run_video_gui(gui)
