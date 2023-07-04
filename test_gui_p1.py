import pytest
import gui_p1
from unittest.mock import MagicMock
from gui_p1 import *
from unittest.mock import patch


@pytest.fixture
def mock_video_capture():
    gui_p1.cap = MagicMock(spec=cv2.VideoCapture)
    gui_p1.cap.isOpened.return_value = True
    gui_p1.cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return gui_p1.cap


def test_parse_arguments():
    with patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(video="path/to/video")):
        gui_p1.video_path = parse_argument()
    assert gui_p1.video_path == "path/to/video"


@pytest.fixture
def gui_p1():
    video_path = "video.mp4"
    gui = VideoGUI(video_path)
    return gui


def test_mouse_callback_add_position(gui_p1):
    gui_p1.ROI_MODE = True
    gui_p1.cursor_positions = []
    gui_p1.mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 200, 0, 0)
    assert gui_p1.cursor_positions == [(100, 200)]


def test_mouse_callback_delete_position(gui_p1):
    gui_p1.ROI_MODE = True
    gui_p1.cursor_positions = [(100, 200)]
    gui_p1.mouse_callback(cv2.EVENT_RBUTTONDOWN, 0, 0, 0, 0)
    assert gui_p1.cursor_positions == []


def test_key_handler_roi_mode_toggle(gui_p1):
    gui_p1.ROI_MODE = False
    cv2.waitKey = lambda _: 32
    gui_p1.key_handler()
    assert gui_p1.ROI_MODE is True


def test_key_handler_det_mode_toggle(gui_p1):
    gui_p1.cursor_positions = [(100, 100), (300, 100), (200, 150)]
    gui_p1.DET_MODE = False
    gui_p1.ROI_MODE = True
    cv2.waitKey = lambda _: 13
    gui_p1.key_handler()
    assert gui_p1.ROI_MODE is False
    assert gui_p1.DET_MODE is True


def test_key_handler_det_mode_save(gui_p1):
    gui_p1.cursor_positions = [(200, 200), (600, 200), (400, 300)]
    gui_p1.saved_convex_shape = []
    gui_p1.DET_MODE = False
    gui_p1.ROI_MODE = True
    cv2.waitKey = lambda _: 13
    gui_p1.key_handler()
    assert gui_p1.saved_convex_shape == [(200, 200), (600, 200), (400, 300)]
    assert gui_p1.saved is True


def test_key_handler_exit(gui_p1):
    cv2.waitKey = lambda _: 27
    result = gui_p1.key_handler()
    assert result == "exit"


@pytest.fixture
def cur_frame():
    return np.zeros((500, 500, 3), dtype=np.uint8)


@pytest.fixture
def expected_frame():
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 5, (255, 0, 0), -1)
    cv2.circle(frame, (200, 200), 5, (255, 0, 0), -1)
    cv2.circle(frame, (300, 300), 5, (255, 0, 0), -1)
    return frame


def test_draw_positions(gui_p1, cur_frame, expected_frame):
    gui_p1.cursor_positions = [(100, 100), (200, 200), (300, 300)]
    gui_p1.draw_positions(cur_frame)
    assert np.array_equal(cur_frame, expected_frame)


def test_draw_connections_one_position(gui_p1):
    cur_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    gui_p1.cursor_positions = [(100, 100)]
    gui_p1.draw_connections(cur_frame)
    expected_frame = np.zeros((640, 480, 3), dtype=np.uint8)

    if len(gui_p1.cursor_positions) > 1:
        for i in range(len(gui_p1.cursor_positions) - 1):
            cv2.line(expected_frame, gui_p1.cursor_positions[i], gui_p1.cursor_positions[i + 1], (212, 255, 127), 2)
        if len(gui_p1.cursor_positions) > 2:
            cv2.line(expected_frame, gui_p1.cursor_positions[-1], gui_p1.cursor_positions[0], (212, 255, 127), 2)

    assert np.array_equal(cur_frame, expected_frame)


def test_draw_connections_two_positions(gui_p1):
    cur_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    gui_p1.cursor_positions = [(100, 100), (200, 200)]
    gui_p1.draw_connections(cur_frame)
    expected_frame = np.zeros((640, 480, 3), dtype=np.uint8)

    if len(gui_p1.cursor_positions) > 1:
        for i in range(len(gui_p1.cursor_positions) - 1):
            cv2.line(expected_frame, gui_p1.cursor_positions[i], gui_p1.cursor_positions[i + 1], (212, 255, 127), 2)
        if len(gui_p1.cursor_positions) > 2:
            cv2.line(expected_frame, gui_p1.cursor_positions[-1], gui_p1.cursor_positions[0], (212, 255, 127), 2)

    assert np.array_equal(cur_frame, expected_frame)


def test_draw_connections_three_positions(gui_p1):
    cur_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    gui_p1.cursor_positions = [(100, 100), (200, 200), (300, 300)]
    gui_p1.draw_connections(cur_frame)
    expected_frame = np.zeros((640, 480, 3), dtype=np.uint8)

    if len(gui_p1.cursor_positions) > 1:
        for i in range(len(gui_p1.cursor_positions) - 1):
            cv2.line(expected_frame, gui_p1.cursor_positions[i], gui_p1.cursor_positions[i + 1], (212, 255, 127), 2)
        if len(gui_p1.cursor_positions) > 2:
            cv2.line(expected_frame, gui_p1.cursor_positions[-1], gui_p1.cursor_positions[0], (212, 255, 127), 2)

    assert np.array_equal(cur_frame, expected_frame)


def test_is_convex_case1():
    positions1 = [(100, 100), (200, 200), (300, 100)]
    result1 = is_convex(positions1)
    assert result1 == cv2.isContourConvex(np.array(positions1))


def test_is_convex_case2():
    positions2 = [(100, 100), (200, 200), (100, 300)]
    result2 = is_convex(positions2)
    assert result2 == cv2.isContourConvex(np.array(positions2))


def test_is_convex_case3():
    positions3 = [(100, 100)]
    result3 = is_convex(positions3)
    assert result3 is False


def test_play_mode(gui_p1, mock_video_capture):
    gui_p1.frame = None
    gui_p1.frame_count = 0
    gui_p1.cap = mock_video_capture

    gui_p1.play_mode()

    assert gui_p1.frame is not None
    assert gui_p1.frame_count == 1


def test_init_video_gui(gui_p1):
    gui_p1.init_video_gui()

    assert gui_p1.cap is not None
    assert isinstance(gui_p1.cap, cv2.VideoCapture)
    assert gui_p1.cap.isOpened()
    assert gui_p1.frame_count == 0
    assert gui_p1.frame is None
    gui_p1.terminate_video_gui()


def test_terminate_video_gui(gui_p1):
    gui_p1.cap = cv2.VideoCapture("video.mp4")
    cv2.namedWindow("GUI")
    gui_p1.terminate_video_gui()
    assert not gui_p1.cap.isOpened()
    assert cv2.getWindowProperty("GUI", cv2.WND_PROP_VISIBLE) == 0


def test_run_video_gui(gui_p1):
    gui_p1.video_path = "video.mp4"
    run_video_gui(gui_p1)

    assert not gui_p1.cap.isOpened()
    assert cv2.getWindowProperty("GUI", cv2.WND_PROP_VISIBLE) == 0
