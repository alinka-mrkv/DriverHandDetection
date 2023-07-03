from video_gui import *
from model_detection import ModelDetection
from plane_detection import PlaneDetection


def main():
    video_path = parse_argument()
    gui = VideoGUI(video_path)
    model_detector = ModelDetection()
    plane_detect = PlaneDetection()
    run_video_gui(gui, model_detector, plane_detect)


if __name__ == "__main__":
    main()
