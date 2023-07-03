import cv2
import numpy as np
from operator import itemgetter
from shapely.geometry import Polygon, Point, LineString
from math import sqrt
from intersection_result import IntersectionResult


class PlaneDetection:
    def __init__(self):
        self.normalized_landmarks = []
        self.world_landmarks = []
        self.driver_px_coordinates = []
        self.world_landmarks_copy = []
        self.intersection_status = False
        self.intersection_vector = ""

    def to_find_vector(self, w_coord_human, n):
        # vector construction from elbow to wrist
        point_elbow = np.array(w_coord_human[n])
        point_wrist = np.array(w_coord_human[n + 2])
        vector = point_wrist - point_elbow
        print("Required vector is" + str(vector))
        return vector, point_elbow

    def to_find_pixel_from_world_check(self, world_coordinates, norm_coordinates, world_points):
        matrix, _ = cv2.findHomography(world_coordinates, norm_coordinates)
        norm_points = []
        for point in world_points:
            w_point = np.float32(np.array([[list(point)]]))
            norm_points.append(
                [
                    cv2.perspectiveTransform(w_point, matrix)[0][0][0],
                    cv2.perspectiveTransform(w_point, matrix)[0][0][1],
                ]
            )
        return 0

    def to_find_world_coord_plane(self, cursor_positions, height, width):
        world_landmarks = []
        norm_landmarks = []
        for landmark in self.world_landmarks:
            for result in landmark:
                world_landmarks.append([result.x, result.y])
        for landmark in self.normalized_landmarks:
            for result in landmark:
                norm_landmarks.append([result.x, result.y])

        world_coordinates = np.array(world_landmarks, dtype=np.float32)
        norm_coordinates = np.array(norm_landmarks, dtype=np.float32)
        matrix, _ = cv2.findHomography(norm_coordinates, world_coordinates)
        normal_c_positions = []
        for point in cursor_positions:
            normal_c_positions.append([point[0] / width, point[1] / height])
        world_points = []
        for point in normal_c_positions:
            norm_point = np.float32(np.array([[list(point)]]))
            world_points.append(
                [
                    cv2.perspectiveTransform(norm_point, matrix)[0][0][0],
                    cv2.perspectiveTransform(norm_point, matrix)[0][0][1],
                ]
            )
        return world_points

    def to_find_all_plane_coord(self, grid, cursor_positions):
        if not cursor_positions:
            return

        x_min = min(i[0] for i in grid)
        x_max = max(i[0] for i in grid)

        y_min = min(i[1] for i in grid)
        y_max = max(i[1] for i in grid)
        plane = [[x_min, y_min, -1], [x_min, y_max, -1], [x_max, y_max, 1], [x_max, y_min, 1]]
        plane = np.asarray(plane)
        return plane

    def to_find_normal(self, grid):
        Сoefficient = np.vstack((grid[0], grid[1], grid[2]))
        Result = np.ones(3)  # Сделали уравнение вида Сoefficient * x = Result
        normal = np.linalg.solve(Сoefficient, Result)
        return normal

    def to_find_vector_to_point_on_plane(self, w_coord_human, grid, n):
        point_on_vector = np.array(w_coord_human[n])
        point_A = grid[0]
        new_vector = point_on_vector - point_A
        return new_vector, point_on_vector

    def to_find_point_m(self, normal, new_vector, vector, point_on_vector):
        dot = normal.dot(vector)
        number = -(normal.dot(new_vector) / dot)
        point_M = point_on_vector + vector * number
        return dot, point_M

    def to_find_point_on_section_1(self, section, point_m):
        point_m = Point(point_m)
        line = LineString(section)
        return point_m.intersects(line)

    def to_find_point_on_section(self, point_m, coord_1, coord_2):
        vector_m_1 = point_m - coord_1
        vector_1_2 = coord_2 - coord_1
        dot_product = vector_1_2.dot(vector_m_1)
        vector_1_2_length = sqrt(vector_1_2[0] ** 2 + vector_1_2[1] ** 2 + vector_1_2[2] ** 2)
        vector_m_1_length = sqrt(vector_m_1[0] ** 2 + vector_m_1[1] ** 2 + vector_m_1[2] ** 2)
        cos = dot_product / (vector_1_2_length * vector_m_1_length)
        value = coord_1[0] <= point_m[0] <= coord_2[0]
        return np.isclose(cos, 1.0) and value

    def to_find_intersection_point(self, w_coord_human, grid, vector, n):
        normal = self.to_find_normal(grid)

        new_vector, point_on_vector = self.to_find_vector_to_point_on_plane(w_coord_human, grid, n)

        dot, point_m = self.to_find_point_m(normal, new_vector, vector, point_on_vector)

        if dot > 0:
            if not self.to_find_point_on_section(point_m, np.array(w_coord_human[n]), np.array(w_coord_human[n + 2])):
                print("The hand doesn't intersect the pseudo-window - from point check")
                return None
            elif not Polygon(grid).contains(Point(point_m)):
                print("The hand doesn't intersect the pseudo-window - from polygon check")
                return None
            else:
                print("Intersection point is " + str(point_m))
                self.intersection_status = True
                if n == 13:
                    self.intersection_vector = "elbow-wrist"
                else:
                    self.intersection_vector = "sholder-elbow"
                return point_m
        else:
            print("The hand doesn't intersect the pseudo-window")
            return None

    def result_of_finding_intersection(self, cursor_positions, height, width):
        if not cursor_positions:
            return
        if not self.world_landmarks:
            return

        w_coord_human = []
        for result in self.world_landmarks:
            for landmark in result:
                w_coord_human.append([landmark.x, landmark.y, landmark.z])

        vector_from_elbow, p_elbow = self.to_find_vector(w_coord_human, 13)
        vector_from_shoulder, p_shoulder = self.to_find_vector(w_coord_human, 11)
        if len(self.world_landmarks_copy) > 3:
            grid = self.to_find_world_coord_plane(cursor_positions, height, width)
            plane = self.to_find_all_plane_coord(grid, cursor_positions)

            inter_elbow = self.to_find_intersection_point(w_coord_human, plane, vector_from_elbow, 13)
            inter_shoulder = self.to_find_intersection_point(w_coord_human, plane, vector_from_shoulder, 11)
            return IntersectionResult(
                p_elbow,
                p_shoulder,
                plane,
                w_coord_human,
                inter_elbow,
                inter_shoulder,
                vector_from_elbow,
                vector_from_shoulder,
            )
        else:
            return None
