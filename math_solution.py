import numpy as np

def triangulation(
    camera_matrix: np.ndarray,
    camera_position1: np.ndarray,
    camera_rotation1: np.ndarray,
    camera_position2: np.ndarray,
    camera_rotation2: np.ndarray,
    image_points1: np.ndarray,
    image_points2: np.ndarray
):
    """
    :param camera_matrix: Intrinsic camera matrix, np.ndarray of shape 3x3
    :param camera_position1: First camera position in world coordinates, np.ndarray of shape 3x1
    :param camera_rotation1: First camera rotation matrix, np.ndarray of shape 3x3
    :param camera_position2: Second camera position in world coordinates, np.ndarray of shape 3x1
    :param camera_rotation2: Second camera rotation matrix, np.ndarray of shape 3x3
    :param image_points1: Points in the first image, np.ndarray of shape Nx2
    :param image_points2: Corresponding points in the second image, np.ndarray of shape Nx2
    :return: Triangulated 3D points, np.ndarray of shape Nx3
    """
    # Transpose rotation matrices for world to camera coordinate transformation
    rot1_transposed = camera_rotation1.T
    rot2_transposed = camera_rotation2.T

    # Compute translation vectors in the camera coordinate system
    t1 = -rot1_transposed @ camera_position1
    t2 = -rot2_transposed @ camera_position2

    # Construct projection matrices
    P1 = camera_matrix @ np.hstack((rot1_transposed, t1))
    P2 = camera_matrix @ np.hstack((rot2_transposed, t2))

    # Triangulate points
    points_3d = []
    for p1, p2 in zip(image_points1, image_points2):
        A = np.array([
            p1[0] * P1[2] - P1[0],
            p1[1] * P1[2] - P1[1],
            p2[0] * P2[2] - P2[0],
            p2[1] * P2[2] - P2[1]
        ])

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]  # Homogeneous to Cartesian conversion
        points_3d.append(X[:3])

    return np.array(points_3d)
