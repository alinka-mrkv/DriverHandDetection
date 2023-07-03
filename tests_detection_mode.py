import pytest
import numpy as np

from class_detection_model import ModelDetection
from class_finding_coordinates import PlaneDetection


@pytest.fixture
def create_object_of_model_class():
    model_detector = ModelDetection()
    return model_detector


@pytest.fixture
def create_object_of_plane_class():
    plane_detect = PlaneDetection()
    plane_detect.world_landmarks = []
    plane_detect.normalized_landmarks = []
    plane_detect.intersection_status = False
    plane_detect.intersection_vector = ""
    return plane_detect

@pytest.fixture
def create_human_world_landmarks():
    w_coord_human = [[-0.33118540048599243, -0.547090470790863, -0.21993939578533173], 
                    [-0.3098369538784027, -0.5712825059890747, -0.23961423337459564],
                    [-0.3087897300720215, -0.5718857049942017, -0.22808752954006195], 
                    [-0.31061309576034546, -0.57108473777771, -0.23043635487556458], 
                    [-0.3179134726524353, -0.578557550907135, -0.22537104785442352], 
                    [-0.31661975383758545, -0.5775521993637085, -0.23827047646045685], 
                    [-0.307925820350647, -0.5677003860473633, -0.21803346276283264], 
                    [-0.1921217441558838, -0.5755002498626709, -0.23956890404224396], 
                    [-0.23308488726615906, -0.5190470218658447, -0.11694245040416718], 
                    [-0.2693045139312744, -0.5432827472686768, -0.20765894651412964], 
                    [-0.2758147716522217, -0.5085214376449585, -0.20021559298038483], 
                    [-0.06759525835514069, -0.46267011761665344, -0.176699697971344], 
                    [-0.18602587282657623, -0.48022544384002686, 0.05880814790725708], 
                    [-0.06650035083293915, -0.4112474024295807, -0.272639662027359], 
                    [-0.16365820169448853, -0.2996293306350708, 0.04433128982782364], 
                    [-0.18352708220481873, -0.38477838039398193, -0.25437992811203003], 
                    [-0.2269834578037262, -0.2449115365743637, -0.07188207656145096], 
                    [-0.2203495055437088, -0.38701504468917847, -0.256799578666687], 
                    [-0.21356201171875, -0.2506186068058014, -0.11201970279216766], 
                    [-0.23253732919692993, -0.396839439868927, -0.25909653306007385],
                    [-0.20995062589645386, -0.2934146523475647, -0.1290871500968933], 
                    [-0.18108800053596497, -0.37513479590415955, -0.25532492995262146], 
                    [-0.22947829961776733, -0.25818267464637756, -0.08763766288757324], 
                    [0.035675592720508575, -0.013660099357366562, -0.030629264190793037], 
                    [-0.03509228304028511, 0.009205964393913746, 0.03222757205367088], 
                    [0.002276260405778885, 0.19193768501281738, -0.0615038126707077], 
                    [0.0052385106682777405, 0.29306888580322266, 0.04370824620127678], 
                    [0.09132175147533417, 0.4887135326862335, 0.12341056019067764], 
                    [0.014368966221809387, 0.5440780520439148, 0.2172836810350418], 
                    [0.07934293150901794, 0.5231472849845886, 0.1506933867931366], 
                    [0.011230528354644775, 0.5611832737922668, 0.19779479503631592], 
                    [0.054594486951828, 0.5761333703994751, 0.14277255535125732], 
                    [-0.018839016556739807, 0.5833703875541687, 0.17962002754211426]]
    return w_coord_human


'''
def test_to_find_world_coord_plane(create_object_of_plane_class):
    cursor_positions = [(968, 138), (1100, 136), (1147, 605), (1000, 566)]
    height = 720
    width = 1280
    normal_c_positions = []
    for point in cursor_positions:
        normal_c_positions.append([point[0] / width, point[1] / height])
    world_points = create_object_of_plane_class.to_find_world_coord_plane(cursor_positions, height, width)
    check_world_points = create_object_of_plane_class.to_find_pixel_from_world_check(create_object_of_plane_class.world_landmarks, create_object_of_plane_class.normalized_landmarks, world_points)
    normal_c_positions = [round(i, 1) for i in normal_c_positions]
    check_world_points = [round(i, 1) for i in check_world_points]
    assert np.array_equal(normal_c_positions, check_world_points)
'''

def test_to_find_vector(create_object_of_plane_class, create_human_world_landmarks):
    vector_elbow, p_elbow = create_object_of_plane_class.to_find_vector(create_human_world_landmarks, 13)
    vector_shoulder, p_sholder = create_object_of_plane_class.to_find_vector(create_human_world_landmarks, 11)
    assert vector_elbow.all() == np.asarray([-0.11702673, 0.02646902, 0.01825973]).all()
    assert p_elbow.all() == np.asarray([-0.06650035, -0.4112474, -0.27263966]).all()
    assert vector_shoulder.all() == np.asarray([0.00109491, 0.05142272, -0.09593996]).all()
    assert p_sholder.all()  == np.asarray([-0.06759526, -0.46267012, -0.1766997]).all()


def test_to_find_all_plane_coord(create_object_of_plane_class):
    cursor_positions = [(968, 138), (1100, 136), (1147, 605), (1000, 566)]
    grid = [[-1.7702537, 5.429199], [-1.8248407, 5.5787477], [-1.9043047, 5.0600142], [-1.8577878, 4.897764]]
    plane = create_object_of_plane_class.to_find_all_plane_coord(grid, cursor_positions)
    assert len(plane) != 0


def test_to_find_all_plane_coord_1(create_object_of_plane_class):
    сursor_positions = [(968, 138), (1100, 136), (1147, 605), (1000, 566)]
    grid = [[-1.7702537, 5.429199], [-1.8248407, 5.5787477], [-1.9043047, 5.0600142], [-1.8577878, 4.897764]]
    plane = create_object_of_plane_class.to_find_all_plane_coord(grid, сursor_positions)
    assert (plane[i][0] == -1.7702537 or plane[i][0] == -1.9043047 for i in enumerate(grid))


def test_to_find_all_plane_coord_2(create_object_of_plane_class):
    сursor_positions = [(968, 138), (1100, 136), (1147, 605), (1000, 566)]
    grid = [[-1.7702537, 5.429199], [-1.8248407, 5.5787477], [-1.9043047, 5.0600142], [-1.8577878, 4.897764]]
    plane = create_object_of_plane_class.to_find_all_plane_coord(grid, сursor_positions)
    assert (plane[i][1] == 5.5787477 or plane[i][1] == 4.897764 for i in enumerate(grid))


def test_to_find_all_plane_coord_3(create_object_of_plane_class):
    сursor_positions = [(968, 138), (1100, 136), (1147, 605), (1000, 566)]
    grid = [[-1.7702537, 5.429199], [-1.8248407, 5.5787477], [-1.9043047, 5.0600142], [-1.8577878, 4.897764]]
    plane = create_object_of_plane_class.to_find_all_plane_coord(grid, сursor_positions)
    print(plane)
    assert len(plane) == 4


def test_to_find_normal(create_object_of_plane_class):
    plane = [[2.36501789, -13.98093605, -1. ], [2.36501789, -5.31192255, -1. ], [4.17437315, -5.31192255, 1. ], [4.17437315, -13.98093605, 1. ]]
    normal = create_object_of_plane_class.to_find_normal(plane)
    assert normal.all() == np.asarray([ 0.30583888, -0. , -0.27668559]).all()


def test_to_find_vector_to_point_on_plane(create_object_of_plane_class, create_human_world_landmarks):
    plane = [[2.36501789, -13.98093605, -1. ], [2.36501789, -5.31192255, -1. ], [4.17437315, -5.31192255, 1. ], [4.17437315, -13.98093605, 1. ]]
    vector_13, point_on_vector_13 = create_object_of_plane_class.to_find_vector_to_point_on_plane(create_human_world_landmarks, plane, 13)
    vector_11, point_on_vector_11 = create_object_of_plane_class.to_find_vector_to_point_on_plane(create_human_world_landmarks, plane, 11)
    assert vector_13.all() == np.asarray([-2.43151824, 13.56968865,  0.72736034]).all()
    assert point_on_vector_13.all() == np.asarray([-0.06650035, -0.4112474 , -0.27263966]).all()
    assert vector_11.all() == np.asarray([-2.43261315, 13.51826593, 0.8233003]).all()
    assert point_on_vector_11.all() == np.asarray([-0.06759526, -0.46267012, -0.1766997]).all()


def test_to_find_point_m(create_object_of_plane_class):
    normal = np.array([ 0.30583888, -0. , -0.27668559])
    new_vector = np.asarray([-2.43151824, 13.56968865,  0.72736034])
    point_on_vector = np.array([-0.06650035, -0.4112474 , -0.27263966])
    vector = np.array([-0.11702673, 0.02646902, 0.01825973])
    dot, point_m = create_object_of_plane_class.to_find_point_m(normal, new_vector, vector, point_on_vector)
    assert dot == -0.0408435282015531
    assert point_m.all() == np.asarray([ 2.64087842, -1.02360035, -0.69507311]).all()


def test_to_find_point_on_section(create_object_of_plane_class):
    value = create_object_of_plane_class.to_find_point_on_section(np.array([0.5, 0., 0.]), np.array([0., 0., 0.]), np.array([1., 0., 0.]))
    assert value == True


def test_to_find_point_on_section_1(create_object_of_plane_class):
    value = create_object_of_plane_class.to_find_point_on_section(np.array([-1., 0., 0.]), np.array([0., 0., 0.]), np.array([1., 0., 0.]))
    assert value == False


def test_to_find_point_on_section_2(create_object_of_plane_class):
    value = create_object_of_plane_class.to_find_point_on_section(np.array([0, 1., 0.]), np.array([0., 0., 0.]), np.array([1., 0., 0.]))
    assert value == False


def test_to_find_point_on_section_3(create_object_of_plane_class):
    value = create_object_of_plane_class.to_find_point_on_section(np.array([0., 1., 0.]), np.array([0., 0., 0.]), np.array([0., 1., 0.]))
    assert value == True


def test_to_find_intersection_point(create_object_of_plane_class, create_human_world_landmarks):
    plane = [[2.36501789, -13.98093605, -1. ], [2.36501789, -5.31192255, -1. ], [4.17437315, -5.31192255, 1. ], [4.17437315, -13.98093605, 1. ]]
    vector = np.array([-0.11702673, 0.02646902, 0.01825973])
    point = create_object_of_plane_class.to_find_intersection_point(create_human_world_landmarks, plane, vector, 13)
    assert point is None


