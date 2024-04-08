import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import List

import dask
import dask.array as da
import laspy
import matplotlib
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns
import stack_data
from matplotlib import pyplot as plt
from numba import njit
from scipy.spatial import KDTree

import metashapelib as mslib
from thirdparty import transformations as tf

matplotlib.use("qt5agg")


logger = mslib.getlogger(name="metashapelib", log_level="INFO")


def load_pcd(pcd_path: Path) -> np.ndarray:
    return np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points).astype(np.float32)


def write_pcd_las(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rgb: np.ndarray = None,
    offset=np.array([0.0, 0.0, 0.0]),
    precision: float = 1e-5,
    **kwargs,
) -> bool:
    """
    Write point cloud data to a LAS file.

    Args:
        path (Path): The path to save the LAS file.
        x (np.ndarray): Array of x-coordinates of the points.
        y (np.ndarray): Array of y-coordinates of the points.
        z (np.ndarray): Array of z-coordinates of the points.
        rgb (np.ndarray, optional): Array of RGB values for each point. Must be a 3xn numpy array of 16-bit unsigned integers. Defaults to None.
        offset (np.ndarray, optional): Offset to be added to the coordinates.Defaults to np.array([0.0, 0.0, 0.0]).
        precision (float, optional): Precision of the coordinates. Defaults to 1e-5.
        **kwargs: Additional keyword arguments to be written as extra scalar fields in the LAS file.

    Returns:
        bool: True if writing to the LAS file is successful, False otherwise.

    Raises:
        TypeError: If the rgb argument is not a 3xn numpy array of 16-bit unsigned integers.

    Note:
        - If the path does not end with '.las', '.las' will be appended to the file name.
        - If the rgb argument is provided, it must be a 3xn numpy array of 16-bit unsigned integers.
        - The header of the LAS file will be set to version "1.4" and point format "2".
        - Additional keyword arguments (**kwargs) will be written as extra scalar fields in the LAS file.
    """
    path = Path(path)
    if path.suffix != ".las":
        path = path.parent / (path.name + ".las")

    path.parent.mkdir(parents=True, exist_ok=True)

    if rgb is not None:
        if not isinstance(rgb, np.ndarray) or rgb.shape[1] != 3:
            raise TypeError(
                "Inval rgb argument. It must a 3xn numpy array of 16bit unsigned int"
            )
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.uint16)
        elif rgb.dtype != np.uint16:
            raise TypeError(
                "Inval rgb argument. It must a 3xn numpy array of 16bit unsigned int"
            )

    header = laspy.LasHeader(version="1.4", point_format=3)
    header.add_extra_dims(
        [laspy.ExtraBytesParams(k, type=np.float32) for k in kwargs.keys()]
    )
    header.offsets = offset
    header.scales = [
        precision for _ in range(3)
    ]  # This is the precision of the coordinates

    with laspy.open(path, mode="w", header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(x.shape[0], header=header)
        point_record.x = x.astype(np.float32)
        point_record.y = y.astype(np.float32)
        point_record.z = z.astype(np.float32)
        if rgb is not None:
            point_record.red = rgb[:, 0]
            point_record.green = rgb[:, 1]
            point_record.blue = rgb[:, 2]

        for k, v in kwargs.items():
            setattr(point_record, k, v)

        writer.write_points(point_record)

    return True


def write_pcd(
    xyz: np.ndarray,
    path: Path,
) -> bool:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(str(path), pcd)
    return True


def lazy_load_pcd_stack(pcd_list: List[Path]):
    # load_pcd the first pcd (assume rest are same shape/dtype)
    sample = load_pcd(pcd_list[0])

    # Build a list of lazy dask arrays
    arrays = [
        da.from_delayed(
            dask.delayed(load_pcd)(path),
            dtype=sample.dtype,
            shape=sample.shape,
        )
        for path in pcd_list
    ]
    # Stack all point coordinates into a 3D dask array
    stack = da.stack(arrays, axis=0)
    stack = stack.rechunk()
    return stack


def load_pcd_stack(pcd_list: List[Path]):
    arrays = [load_pcd(path) for path in pcd_list]
    stack = np.stack(arrays, axis=0)
    return stack
