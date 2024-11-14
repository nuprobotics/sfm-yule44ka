from cv2_solution import get_matches, convert_to_world_frame, triangulation, resection
from math_solution import triangulation as math_triangulation
import cv2
import numpy as np
import unittest
import yaml
import ast
import json
import base64


def dict_to_keypoints(keypoints_dict):
    return [
        cv2.KeyPoint(
            x=kp["pt"][0],
            y=kp["pt"][1],
            size=kp["size"],
            angle=kp["angle"],
            response=kp["response"],
            octave=kp["octave"],
            class_id=kp["class_id"]
        )
        for kp in keypoints_dict
    ]


def dict_to_matches(matches_dict):
    return [
        cv2.DMatch(
            _queryIdx=match["queryIdx"],
            _trainIdx=match["trainIdx"],
            _distance=match["distance"],
            _imgIdx=match["imgIdx"]
        )
        for match in matches_dict
    ]


class MathTriangulationTest(unittest.TestCase):

    def test_opencv_usage(self):
        with open("math_solution.py", 'r') as file:
            tree = ast.parse(file.read(), filename="math_solution.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'cv2':
                        self.assertTrue(False, "You should not use OpenCV in math_solution.py")

            elif isinstance(node, ast.ImportFrom):
                if node.module == 'cv2':
                    self.assertTrue(False, "You should not use OpenCV in math_solution.py")

    def test_math_triangulation(self):
        arrays = np.load('tests_assets/math_triangulation_data.npz')
        camera_position = arrays['camera_position']
        camera_rotation = arrays['camera_rotation']
        key_points_1 = arrays['keypoints_1']
        key_points_2 = arrays['keypoints_2']
        correct_triangulation = arrays['correct_triangulation']
        wrong_triangulation = arrays['wrong_triangulated_points']
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        camera_matrix = np.array(config["camera_matrix"], dtype=np.float32, order='C')
        points_3d = math_triangulation(
            camera_matrix,
            np.array([[0],[0],[0]]),
            np.eye(3),
            camera_position,
            camera_rotation,
            key_points_1,
            key_points_2
        )
        if not np.allclose(points_3d, correct_triangulation):
            if np.allclose(points_3d, wrong_triangulation):
                self.assertTrue(False, "Make sure you've interpreted camera position and camera rotation as world coordinates")
            else:
                self.assertTrue(False, "Triangulation is incorrect. Expected: \n{}\nGot: \n{}".format(correct_triangulation, points_3d))



class GetMatchesTest(unittest.TestCase):
    @staticmethod
    def matches_to_dict(matches):
        return [
            {
                "queryIdx": match.queryIdx,   # Индекс ключевой точки в первом изображении
                "trainIdx": match.trainIdx,   # Индекс ключевой точки во втором изображении
                "distance": match.distance,    # Расстояние между точками
                "imgIdx": match.imgIdx        # Индекс изображения`
            }
            for match in matches
        ]

    @staticmethod
    def check_match(match, gt_matches, keypoints1, keypoints2, gt_keypoints1, gt_keypoints2):
        for m in gt_matches:
            if match.queryIdx == m.queryIdx and match.trainIdx == m.trainIdx:
                kp1 = keypoints1[match.queryIdx]
                kp2 = keypoints2[match.trainIdx]
                gt_kp1 = gt_keypoints1[m.queryIdx]
                gt_kp2 = gt_keypoints2[m.trainIdx]
                if kp1.pt == gt_kp1.pt and kp2.pt == gt_kp2.pt:
                    return True
        return False

    def test_get_matches(self):
        image1 = cv2.imread('./images/image0.jpg')
        image2 = cv2.imread('./images/image1.jpg')
        tested_kp1, tested_kp2, tested_matches = get_matches(image1, image2)
        with open("./tests_assets/first_match", "r") as file:
            base64_str = file.read()

        json_str = base64.b64decode(base64_str).decode('utf-8')

        data = json.loads(json_str)

        gt_keypoints1 = dict_to_keypoints(data["keypoints1"])
        gt_keypoints2 = dict_to_keypoints(data["keypoints2"])
        gt_matches = dict_to_matches(data["matches"])

        self.assertEqual(len(tested_kp1), len(gt_keypoints1))
        self.assertEqual(len(tested_kp2), len(gt_keypoints2))
        self.assertEqual(len(tested_matches), len(gt_matches))

        for match in tested_matches:
            self.assertTrue(
                self.check_match(match, gt_matches, tested_kp1, tested_kp2, gt_keypoints1, gt_keypoints2),
                f"All matches should be in ground truth matches, your match:"
                            f"\n{self.matches_to_dict([match])}"
                            f"\nGround truth matches:\n{self.matches_to_dict(gt_matches)}"
            )


class TriangulationTest(unittest.TestCase):
    def test_triangulation(self):
        arrays = np.load('tests_assets/math_triangulation_data.npz')
        translation_vector = arrays['translation_vector']
        rotation_matrix = arrays['rotation_matrix']
        correct_triangulation = arrays['correct_triangulation']
        wrong_triangulation = arrays['wrong_triangulated_points']
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        camera_matrix = np.array(config["camera_matrix"], dtype=np.float32, order='C')
        with open("./tests_assets/first_match", "r") as file:
            base64_str = file.read()

        json_str = base64.b64decode(base64_str).decode('utf-8')

        data = json.loads(json_str)

        gt_keypoints1 = dict_to_keypoints(data["keypoints1"])
        gt_keypoints2 = dict_to_keypoints(data["keypoints2"])
        gt_matches = dict_to_matches(data["matches"])
        points_3d = triangulation(
            camera_matrix,
            np.array([[0],[0],[0]]),
            np.eye(3),
            translation_vector,
            rotation_matrix,
            gt_keypoints1,
            gt_keypoints2,
            gt_matches
        )
        if not np.allclose(points_3d, correct_triangulation):
            if np.allclose(points_3d, wrong_triangulation):
                self.assertTrue(False, "Make sure you've interpreted camera position and camera rotation as world coordinates")
            else:
                self.assertTrue(False, "Triangulation is incorrect. Expected: \n{}\nGot: \n{}".format(correct_triangulation, points_3d))



class ResectionTest(unittest.TestCase):
    def test_resection(self):
        image1 = cv2.imread('./images/image0.jpg')
        image2 = cv2.imread('./images/image2.jpg')

        with open("./tests_assets/first_match", "r") as file:
            base64_str = file.read()

        json_str = base64.b64decode(base64_str).decode('utf-8')

        data = json.loads(json_str)

        gt_matches = dict_to_matches(data["matches"])

        arrays = np.load('tests_assets/math_triangulation_data.npz')
        correct_triangulation = arrays['correct_triangulation']


        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        camera_matrix = np.array(config["camera_matrix"])

        rvec, tvec = resection(image1, image2, camera_matrix, gt_matches, correct_triangulation)

        resection_arrays = np.load('tests_assets/resection.npz')
        r_matrix = resection_arrays["r_matrix"]
        t_vector = resection_arrays["t_vector"]

        self.assertTrue(rvec.shape == r_matrix.shape)
        self.assertTrue(tvec.shape == t_vector.shape)
        self.assertTrue(np.allclose(rvec, r_matrix))
        self.assertTrue(np.allclose(tvec, t_vector))


class ConvertToWorldFrameTest(unittest.TestCase):

    @staticmethod
    def check_result(test_sample_t, test_sample_r, gt_position, gt_rotation):
        camera_position, camera_rotation = convert_to_world_frame(test_sample_t, test_sample_r)
        if np.allclose(camera_position, gt_position) and np.allclose(camera_rotation, gt_rotation):
            return True
        return False


    def test_convert_to_world_frame(self):
        arrays = np.load("tests_assets/convert_to_world_frame_data.npz")
        test_sample_r_1 = arrays["test_sample_r_1"]
        test_sample_t_1 = arrays["test_sample_t_1"]
        test_sample_r_1_result = arrays["test_sample_r_1_result"]
        test_sample_t_1_result = arrays["test_sample_t_1_result"]
        test_sample_r_2 = arrays["test_sample_r_2"]
        test_sample_t_2 = arrays["test_sample_t_2"]
        test_sample_r_2_result = arrays["test_sample_r_2_result"]
        test_sample_t_2_result = arrays["test_sample_t_2_result"]
        test_sample_r_3 = arrays["test_sample_r_3"]
        test_sample_t_3 = arrays["test_sample_t_3"]
        test_sample_r_3_result = arrays["test_sample_r_3_result"]
        test_sample_t_3_result = arrays["test_sample_t_3_result"]
        test_sample_r_3_wrong_result = arrays["test_sample_r_3_wrong_result"]
        test_sample_t_3_wrong_result = arrays["test_sample_t_3_wrong_result"]
        self.assertTrue(self.check_result(test_sample_t_1, test_sample_r_1, test_sample_t_1_result, test_sample_r_1_result))
        self.assertTrue(self.check_result(test_sample_t_2, test_sample_r_2, test_sample_t_2_result, test_sample_r_2_result))
        camera_position, camera_rotation = convert_to_world_frame(test_sample_t_3, test_sample_r_3)
        if np.allclose(camera_position, test_sample_t_3_wrong_result) and np.allclose(camera_rotation, test_sample_r_3_wrong_result):
            self.assertTrue(False, "Make sure you're using transpose instead of inverse matrix")
        self.assertTrue(self.check_result(test_sample_t_3, test_sample_r_3, test_sample_t_3_result, test_sample_r_3_result))


