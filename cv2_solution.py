import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def get_matches(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher()
    matches1 = [m for m, n in bf.knnMatch(des1, des2, k=2) if m.distance < 0.75 * n.distance]
    matches2 = [m for m, n in bf.knnMatch(des2, des1, k=2) if m.distance < 0.75 * n.distance]

    good_matches = [m for m in matches1 if any(m.trainIdx == n.queryIdx and m.queryIdx == n.trainIdx for n in matches2)]
    return kp1, kp2, good_matches

def estimate_pose(kp1, kp2, matches, camera_matrix):
    points1 = np.array([kp1[m.queryIdx].pt for m in matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in matches])
    E, _ = cv2.findEssentialMat(points1, points2, camera_matrix)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)
    return R, t

def triangulate_points(camera_matrix, R, t, kp1, kp2, matches):
    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = camera_matrix @ np.hstack((R, t))

    points1 = np.array([kp1[m.queryIdx].pt for m in matches]).T
    points2 = np.array([kp2[m.trainIdx].pt for m in matches]).T

    points_3d = cv2.triangulatePoints(P1, P2, points1, points2).T
    return (points_3d / points_3d[:, 3:4])[..., :3]

def visualize(camera_positions, camera_rotations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'red', 'green']

    for i, (pos, rot) in enumerate(zip(camera_positions, camera_rotations)):
        ax.scatter(*pos.flatten(), color=colors[i], s=100)
        ax.quiver(*pos.flatten(), *rot[:, 2], length=1, color='magenta')

    ax.set_xlim([-1.5, 2.0])
    ax.set_ylim([-0.5, 3.0])
    ax.set_zlim([-0.5, 3.0])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def main():
    img1 = cv2.imread('./images/image0.jpg')
    img2 = cv2.imread('./images/image1.jpg')
    img3 = cv2.imread('./images/image2.jpg')

    with open("config.yaml", "r") as file:
        camera_matrix = np.array(yaml.safe_load(file)["camera_matrix"], dtype=np.float32)

    kp1, kp2, matches = get_matches(img1, img2)
    R2, t2 = estimate_pose(kp1, kp2, matches, camera_matrix)
    points_3d = triangulate_points(camera_matrix, R2, t2, kp1, kp2, matches)

    R3, t3 = estimate_pose(kp1, kp2, matches, camera_matrix)
    visualize([np.zeros((3, 1)), t2, t3], [np.eye(3), R2, R3])

if __name__ == "__main__":
    main()
