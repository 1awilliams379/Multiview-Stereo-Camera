import cv2
import torch
import time
import numpy as np
import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
import copy

Q = np.array(([1.0, 0.0, 0.0, -160.0],
              [0.0, 1.0, 0.0, -120.0],
              [0.0, 0.0, 0.0, 350.0],
              [0.0, 0.0, 1.0/90.0, 0.0]),dtype=np.float32)

print(torch.version.cuda)
print(torch.__version__)
model_type = "DPT_Hybrid"
#model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    print("cuda")
else:
    print("dud")

midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cap = cv2.VideoCapture('http://10.0.0.107/cam-hi.jpg')

pointCloudArray = []
num = 0

if not cap.isOpened():
    print("Error opening video file")

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
while True:

    succes, img = cap.read()
    print(succes)
    start = time.time()
    while succes == False:
        cap.release()
        cap = cv2.VideoCapture('http://10.0.0.107/cam-hi.jpg')
        succes, img = cap.read()
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

    mask_map = depth_map > 0.4

    output_points = points_3D[mask_map]
    output_colors = img[mask_map]

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    slam = o3d.pipelines.slam.SLAM()




    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow("Image" , img)
    cv2.imshow('Depth Map' , depth_map)

    if cv2.waitKey(5) & 0xFF == 27:
        break

    def create_output(vertices, colors, filename):
        colors = colors.reshape(-1,3)
        vertices = np.hstack([vertices.reshape(-1,3),colors])

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
            f.write(ply_header %dict(vert_num=len(vertices)))
            np.savetxt(f,vertices,'%f %f %f %d %d %d')
    

    output_file = 'clouds/pointCloudDeepLearning' + str(num) + '.ply'
    pcd = o3d.io.read_point_cloud("C:/Users/U/Documents/Python/Esp32/clouds/pointCloudDeepLearning" + str(num) + ".ply")
    num+=1
    create_output(output_points, output_colors, output_file)
    pointCloudArray.append(pcd)



# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()

print(pointCloudArray)



source = o3d.io.read_point_cloud("C:/Users/U/Documents/Python/Esp32/clouds/pointCloudDeepLearning0.ply")
for cloud in pointCloudArray:
    target = cloud
    #combined_pcd = combined_pcd + cloud
    #combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.05)

#o3d.visualization.draw_geometries([source])
    
pcd = o3d.io.read_point_cloud("C:/Users/U/Documents/Python/Esp32/pointCloudDeepLearning0.ply")
o3d.visualization.draw_geometries(pcd)