import os

import numpy as np
import torch
import trimesh
import open3d as o3d
import random


def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def concat_home_dir(path):
    return os.path.join(os.environ['HOME'], 'data', path)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj


def sample_from_omega(bbox, num_points):
    box_points = np.asarray(bbox.get_box_points())

    min_x, max_x = np.min(box_points[:, 0], axis=0), np.max(box_points[:, 0], axis=0)
    min_y, max_y = np.min(box_points[:, 1], axis=0), np.max(box_points[:, 1], axis=0)
    min_z, max_z = np.min(box_points[:, 2], axis=0), np.max(box_points[:, 2], axis=0)

    sample = [np.array([random.uniform(min_x, max_x), random.uniform(min_y, max_y),
                        random.uniform(min_z, max_z)]) for i in range(num_points)]

    return sample


def visualize_pcd(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set)
    o3d.visualization.draw_geometries([pcd], window_name="test")


def load_point_cloud_by_file_extension(file_name, normalize=False, visualize_pointset=True):
    ext = file_name.split('.')[-1]

    if ext == "npz" or ext == "npy":
        point_set = np.load(file_name)
    else:
        point_set = np.asarray(trimesh.load(file_name, ext).vertices)

    if normalize:
        max_norm = max(np.linalg.norm(point_set, axis=1))
        point_set /= max_norm

    if visualize_pointset:
        visualize_pcd(point_set)

    return torch.tensor(point_set).float()


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)
