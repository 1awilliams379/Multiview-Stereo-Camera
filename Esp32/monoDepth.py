import cv2
import torch
import time
import numpy as np
import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
import copy

# Q Matrix for reprojecting depth to 3D 
Q = np.array(([1.0, 0.0, 0.0, -160.0],
              [0.0, 1.0, 0.0, -120.0],
              [0.0, 0.0, 0.0, 350.0],
              [0.0, 0.0, 1.0/90.0, 0.0]), dtype=np.float32)

# Load MiDaS Model
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")

# Options: "DPT_Large" (best), "DPT_Hybrid" (balanced), "MiDaS_small" (fastest)
model_type = "DPT_Hybrid"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Use GPU if available (10-50x faster than CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    print("Using CUDA GPU acceleration")
else:
    print("CUDA not available, using CPU")

midas.to(device)
midas.eval()  # Switch to inference mode (faster, consistent outputs)

# Load image preprocessor - resizes and normalizes images for the model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Camera Setup
ESP32_URL = 'http://10.0.0.107/cam-hi.jpg'
cap = cv2.VideoCapture(ESP32_URL)

pointCloudArray = []
num = 0

if not cap.isOpened():
    print("Error: Unable to connect to ESP32-CAM")
    sys.exit(1)


def preprocess_point_cloud(pcd, voxel_size):
    """
    Prepare point cloud for registration: downsample, compute normals, compute FPFH features.
    FPFH features are 33-number "fingerprints" that help match similar surfaces.
    """
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Normals = vectors pointing perpendicular to surface at each point
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # FPFH = Fast Point Feature Histograms (describes local surface shape)
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def draw_registration_result(source, target, transformation):
    """
    Visualize registration: source=orange, target=blue.
    Good alignment = colors overlap.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996])


def create_output(vertices, colors, filename):
    """Save colored point cloud to PLY file (viewable in MeshLab, CloudCompare)."""
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


# Main Processing Loop
while True:
    success, img = cap.read()
    start = time.time()

    # Reconnect if frame capture failed
    if not success:
        print("Frame capture failed, reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(ESP32_URL)
        success, img = cap.read()
        if not success:
            continue

    # Convert BGR→RGB (OpenCV uses BGR, PyTorch expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess and run depth estimation
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        # Resize depth map back to original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy and normalize to 0-1 range
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Convert 2D+depth to 3D coordinates using Q matrix
    points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

    # Filter unreliable depth values (keep only depth > 0.4)
    mask_map = depth_map > 0.4
    output_points = points_3D[mask_map]
    output_colors = img[mask_map]

    # Calculate FPS
    end = time.time()
    fps = 1 / (end - start)

    # Prepare display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Colorize depth map (MAGMA: purple=close, yellow=far)
    depth_display = (depth_map * 255).astype(np.uint8)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.imshow('Depth Map', depth_display)

    # ESC to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Save point cloud
    output_file = f'clouds/pointCloudDeepLearning{num}.ply'
    create_output(output_points, output_colors, output_file)

    pcd = o3d.io.read_point_cloud(output_file)
    pointCloudArray.append(pcd)
    num += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"Captured {len(pointCloudArray)} point clouds")

if len(pointCloudArray) > 0:
    source = o3d.io.read_point_cloud("clouds/pointCloudDeepLearning0.ply")
    o3d.visualization.draw_geometries([source])
