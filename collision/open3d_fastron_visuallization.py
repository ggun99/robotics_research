import numpy as np
import pandas as pd
import open3d as o3d

# # 엑설에 data 형식으로 저장한 것을 불러오는 코드
# df = pd.read_excel('x0_coll_dataset.xlsx')

# # 값들만 불러오는
# data = df.values

# # pointcloud 만들 때 필요한 xyz값만 가져옴
# xyz = data[:,1:4]

# # pointcloud data 만들어서 저장하는 과정
# pcd = o3d.geometry.PointCloud()
# pcd.points =o3d.utility.Vector3dVector(xyz)
# o3d.io.write_point_cloud("collision_x0.ply", pcd)

# # pointcloud 불러와서 보여주는
col = o3d.io.read_point_cloud("collision_x0.ply")

# # pointcloud 색상을 지정하는 코드 밑의 색은 노란색
coll = col.paint_uniform_color([1, 0.706, 0])
coll.estimate_normals()
# # bounding box 그려줄 수 있음.
aabb = col.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=100, origin=[0, 0, 0])

# # 그리는 코드
# o3d.visualization.draw_geometries([col, mesh_frame])
o3d.visualization.draw([col, mesh_frame])

# # 시점을 저장하려면 ctrl + c 해서 붙여넣으면 시점이랑 다 나옴.
