import open3d as o3d
import torch
import numpy as np
from chamferdist import ChamferDistance
import plotly.express as px
import pandas as pd


def visualize(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    mesh_target = o3d.io.read_triangle_mesh(
        "/home/dhruv/Desktop/IGR-master/dataset/meshes/armadillo.obj")
    mesh_source = o3d.io.read_triangle_mesh(
        "/home/dhruv/Desktop/IGR-master/exps/single_shape/2022_01_25_11_12_47/evaluation/5000/igr_5000_single_shape.ply")

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = mesh_target.vertices
    pcd_target = torch.Tensor(np.array(pcd_target.points)).cuda().unsqueeze(0)

    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = mesh_source.vertices
    pcd_source = torch.Tensor(np.array(pcd_source.points)).cuda().unsqueeze(0)

    df = pd.DataFrame(pcd_target.cpu().numpy()[0, :500, :], columns=['x', 'y', 'z'])
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='x')
    fig.show()

    chamferDist = ChamferDistance().cuda()

    dist_forward = chamferDist(pcd_source, pcd_target)
    print(dist_forward.detach().cpu().item())

    # o3d.visualization.draw_geometries([pcd], window_name="test")
    # visualize(mesh)
