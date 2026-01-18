import os
import numpy as np
import cv2
import sympy

from tqdm import tqdm
import time

from matplotlib import pyplot as plt

import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.rotations as pr
import pytransform3d.camera as pc

from cycler import cycle


class CameraPoses():

    def __init__(self, data_dir, skip_frames, intrinsic):
        # K = intrinsic matrix (camera's internal properties: focal length, center)
        self.K = intrinsic

        # Starting extrinsic = identity (camera at origin, no rotation)
        self.extrinsic = np.array(((1,0,0,0),(0,1,0,0),(0,0,1,0)))

        # Projection matrix P = K @ [R|t] (converts 3D points to 2D pixels)
        self.P = self.K @ self.extrinsic

        # ORB feature detector - finds up to 3000 distinctive points per image
        self.orb = cv2.ORB_create(3000)

        # FLANN matcher with LSH for binary descriptors (fast approximate matching)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.world_points = []
        self.current_pose = None

    @staticmethod
    def _load_images(filepath, skip_frames):
        """Load images from directory."""
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = []

        for path in tqdm(image_paths[::skip_frames]):
            img = cv2.imread(path)
            if img is not None:
                images.append(img)

        return images


    @staticmethod
    def _form_transf(R, t):
        """Create 4x4 transformation matrix from rotation R and translation t."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_world_points(self):
        return np.array(self.world_points)

    def get_matches(self, img1, img2):
        """
        Find matching features between two frames.
        Returns corresponding point pairs (q1, q2) after Lowe's ratio test.
        """
        # Detect ORB keypoints and compute descriptors
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        # Need at least 6 points for Essential Matrix
        if len(kp1) > 6 and len(kp2) > 6:
            # Find 2 nearest matches for each feature (for ratio test)
            matches = self.flann.knnMatch(des1, des2, k=2)

            # Lowe's ratio test: good match should be much better than 2nd best
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.5 * n.distance:
                        good_matches.append(m)
            except ValueError:
                pass

            # Extract pixel coordinates of matched points
            q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            return q1, q2
        else:
            return None, None

    def get_pose(self, q1, q2):
        """
        Compute camera transformation from matched points.
        Returns 4x4 transformation matrix.
        """
        # Essential Matrix encodes rotation + translation between views
        E, mask = cv2.findEssentialMat(q1, q2, self.K)

        # Decompose E into rotation R and translation t
        R, t = self.decomp_essential_mat_old(E, q1, q2)

        transformation_matrix = self._form_transf(R, np.squeeze(t))

        return transformation_matrix


    def decomp_essential_mat(self, E, q1, q2):
        """Decompose Essential Matrix - newer version (not currently used)."""
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]

        K = np.concatenate((self.K, np.zeros((3,1)) ), axis = 1)
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        max = np.argmax(positives)
        if (max == 2):
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            return R2, np.ndarray.flatten(t)


    def decomp_essential_mat_old(self, E, q1, q2):
        """
        Decompose Essential Matrix into rotation and translation.
        Tests all 4 possible R,t combinations, picks the one where
        most triangulated points have positive Z (in front of cameras).
        """
        def sum_z_cal_relative_scale(R, t):
            T = self._form_transf(R, t)
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate 3D points from 2D correspondences
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)

            # Convert from homogeneous coordinates (divide by W)
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Count points with positive Z (in front of camera)
            sum_of_pos_z_Q1 = sum(Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(Q2[2, :] > 0)

            # Calculate relative scale between views
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Get two possible rotations and one translation (±t)
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Four possible combinations
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Test each combination
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Pick the one with most points in front of both cameras
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        # Triangulate final 3D points for storage
        T = self._form_transf(R1, t)
        P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
        hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
        hom_Q2 = np.matmul(T, hom_Q1)
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        self.world_points.append(Q1)

        return [R1, t]


# Load camera intrinsic matrix from calibration file
with open('intrinsicNew.npy', 'rb') as f:
    intrinsic = np.load(f)

# Initialize visual odometry
skip_frames = 2
data_dir = ''
vo = CameraPoses(data_dir, skip_frames, intrinsic)

# Path tracking
gt_path = []
estimated_path = []
camera_pose_list = []

# Starting pose: identity (at origin, no rotation)
start_pose = np.ones((3,4))
start_translation = np.zeros((3,1))
start_rotation = np.identity(3)
start_pose = np.concatenate((start_rotation, start_translation), axis=1)

# Connect to ESP32-CAM
cap = cv2.VideoCapture('http://10.0.0.107/cam-hi.jpg')

if cap.isOpened() == False:
    print("Error opening video stream")

process_frames = False
old_frame = None
new_frame = None
frame_counter = 0
cur_pose = start_pose

# Main processing loop
while(cap.isOpened()):

    ret, new_frame = cap.read()

    # Reconnect if frame capture failed
    while ret == False:
        cap.release()
        cap = cv2.VideoCapture('http://10.0.0.107/cam-hi.jpg')
        ret, new_frame = cap.read()

    frame_counter += 1
    start = time.perf_counter()

    if process_frames and ret:
        # Find matching features between consecutive frames
        q1, q2 = vo.get_matches(old_frame, new_frame)

        if q1 is not None:
            if len(q1) > 20 and len(q2) > 20:
                # Compute transformation and accumulate pose
                transf = vo.get_pose(q1, q2)
                cur_pose = cur_pose @ transf

        # Store pose for visualization
        hom_array = np.array([[0,0,0,1]])
        hom_camera_pose = np.concatenate((cur_pose,hom_array), axis=0)
        camera_pose_list.append(hom_camera_pose)
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))  # X, Z position

        estimated_camera_pose_x, estimated_camera_pose_y = cur_pose[0, 3], cur_pose[2, 3]

    elif process_frames and ret is False:
        break

    old_frame = new_frame
    process_frames = True

    # Calculate FPS
    end = time.perf_counter()
    total_time = end - start
    fps = 1 / total_time

    # Display FPS
    cv2.putText(new_frame, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Display rotation matrix (blue)
    cv2.putText(new_frame, str(np.round(cur_pose[0, 0],2)), (260,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[0, 1],2)), (340,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[0, 2],2)), (420,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[1, 0],2)), (260,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[1, 1],2)), (340,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[1, 2],2)), (420,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[2, 0],2)), (260,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[2, 1],2)), (340,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[2, 2],2)), (420,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    # Display translation vector (red)
    cv2.putText(new_frame, str(np.round(cur_pose[0, 3],2)), (540,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[1, 3],2)), (540,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[2, 3],2)), (540,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("img", new_frame)

    # Press Q to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()

# Visualization
number_of_frames = 20
image_size = np.array([640, 480])

plt.figure()
ax = plt.axes(projection='3d')

camera_pose_poses = np.array(camera_pose_list)

# Plot camera positions in 3D
key_frames_indices = np.linspace(0, len(camera_pose_poses) - 1, number_of_frames, dtype=int)
colors = cycle("rgb")

for i, c in zip(key_frames_indices, colors):
    pc.plot_camera(ax, vo.K, camera_pose_poses[i],
                   sensor_size=image_size, c=c)

plt.show()

# Plot 2D path (bird's eye view)
take_every_th_camera_pose = 2
estimated_path = np.array(estimated_path[::take_every_th_camera_pose])
plt.plot(estimated_path[:,0], estimated_path[:,1])
plt.show()
