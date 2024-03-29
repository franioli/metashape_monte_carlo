import numpy as np

from pathlib import Path
from typing import List

import Metashape

"""Input"""


def read_opencv_calibration(path) -> dict:
    """
    Read camera internal orientation from file, save in camera class
    and return them.
    The file must contain the full K matrix and distortion vector,
    according to OpenCV standards, and organized in one line, as follow:
    width, heigth, fx 0. cx 0. fy cy 0. 0. 1. k1, k2, p1, p2, [k3, [k4, k5, k6
    Values must be float(include the . after integers) and divided by a
    white space.
    -------
    Returns:  cam_prm (dict)
    """
    path = Path(path)
    if not path.exists():
        print("Error: calibration filed does not exist.")
        return None
    with open(path, "r") as f:
        data = np.loadtxt(f)

    if len(data) == 15:
        print("Using OPENCV camera model.")
        dist = data[11:15].astype(float)
    elif len(data) == 16:
        print("Using OPENCV camera model + k3")
        dist = data[11:16].astype(float)
    elif len(data) == 19:
        print("Using FULL OPENCV camera model")
        dist = data[11:19].astype(float)
    else:
        print("invalid intrinsics data.")
        return None
    # TODO: implement other camera models and estimate K from exif.

    cam_prm = {}
    cam_prm["width"] = data[0].astype(int)
    cam_prm["height"] = data[1].astype(int)
    K = data[2:11].astype(float).reshape(3, 3, order="C")
    cam_prm["f"] = K[1, 1]
    cam_prm["cx"] = K[0, 2] - cam_prm["width"] / 2
    cam_prm["cy"] = K[1, 2] - cam_prm["height"] / 2
    cam_prm["k1"] = dist[0]
    cam_prm["k2"] = dist[1]
    cam_prm["p1"] = dist[2]
    cam_prm["p2"] = dist[3]
    if len(dist) > 4:
        cam_prm["k3"] = dist[4]
    else:
        cam_prm["k3"] = 0.0
    if len(dist) > 5:
        cam_prm["k4"] = dist[5]
    else:
        cam_prm["k4"] = 0.0
    cam_prm["b1"] = 0.0
    cam_prm["b2"] = 0.0

    return cam_prm


def read_gcp_file(filename: str, return_raw=False):
    """Read GCPs information from .txt file, organized as in OpenDroneMap
    (https://docs.opendronemap.org/gcp/). The file structure is the following.
    Input file structure:
        <projection>
        geo_x geo_y geo_z im_x im_y image_name [gcp_name] [extra1] [extra2]
    -------
    Return: gcps (List[dict]): List of dictionary containing GCPs info organized by marker, as in "arrange_gcp" function.
    """
    with open(filename, encoding="utf-8") as f:
        data = []
        for line in f:
            l = line.split(" ")
            gcp = {}
            gcp["world"] = np.array([float(x) for x in l[0:3]])
            gcp["projection"] = np.array([float(x) for x in l[3:5]])
            gcp["image"] = l[5:6][0]
            gcp["label"] = l[6:7][0].rstrip()
            data.append(gcp)
    gcps = arrange_gcp(data)
    if return_raw:
        return (gcps, data)
    else:
        return gcps


def find_gcp_in_data(data, label, verbose=False) -> List[dict]:
    """Helpers for collecting together all the projections of the same GCP. It is used by the function arrange_gcp"""
    markers = []
    for line in data:
        if line["label"] == label:
            if verbose:
                print(f'GCP {label} found in image {line["image"]}.')
            markers.append(line)
            continue
    if not markers:
        print(f"GCP {label} not found.")
    return markers


def arrange_gcp(data: dict) -> List[dict]:
    """Reorganize gcp dictionary strcuture (as given by the function read_gcp_file), in a list of dictionaries, each structured hierarchically by GCP label.
    The output structure is the following:
        gcps (list)
            point (dict)
                |-label: str
                |-world: 3x1 np.ndarray -> 3D world coordinates
                |-projections (dict)
                    |- image_name1: str -> image 1 name
                    |- projection1: 2x1 np.ndarray -> projection on image 1
                    |- image_name2: str -> image 2 name
                    |- projection2: 2x1 np.ndarray -> projection on image 2
                    ...
    """

    gcps = []
    for point in data:
        dummy = find_gcp_in_data(data, label=point["label"])
        if point["label"] in [x["label"] for x in gcps]:
            continue
        gcps.append(
            {
                "label": dummy[0]["label"],
                "world": dummy[0]["world"],
                "projections": {x["image"]: tuple(x["projection"]) for x in dummy},
            }
        )
    return gcps


"""Output"""


def write_markers_by_camera(
    chunk: Metashape.Chunk,
    file_name: str,
    convert_to_micron: bool = False,
) -> None:
    """Write Marker image coordinates to csv file,
    sort by camera, as follows:
    cam1, marker1, x, y
    cam1, marker2, x, y
    ...
    cam1, markerM, x, y
    cam2, marker1, x, y
    ....
    camN, markerM, x,Y

    Args:
        chunk (Metashape.Chunk): Metashape Chunk
        file_name (str): path of the output csv file
        convert_to_micron (bool, default = False)
    """

    # Write header to file
    file = open(file_name, "w")

    # If convert_to_micron is True, convert image coordinates from x-y (row,column) image coordinate system to xi-eta image coordinate system (origin at the center of the image, xi towards right, eta upwards)
    if convert_to_micron:
        file.write("image_name,feature_id,xi,eta\n")
    else:
        file.write("image_name,feature_id,x,y\n")

    for camera in chunk.cameras:
        for marker in chunk.markers:
            projections = marker.projections  # list of marker projections
            marker_name = marker.label

            for cur_cam in marker.projections.keys():
                if cur_cam == camera:
                    cam_name = cur_cam.label
                    x, y = projections[cur_cam].coord

                    # writing output to file
                    if convert_to_micron:
                        pixel_size_micron = cur_cam.sensor.pixel_size * 1000
                        image_width = cur_cam.sensor.width
                        image_heigth = cur_cam.sensor.height
                        xi = (x - image_width / 2) * pixel_size_micron[0]
                        eta = (image_heigth / 2 - y) * pixel_size_micron[1]
                        file.write(f"{cam_name},{marker_name:5},{xi:8.1f},{eta:8.1f}\n")
                    else:
                        file.write(f"{cam_name},{marker_name},{x:.4f},{y:.4f}\n")

    file.close()
    print("Marker exported successfully")


def write_markers_by_marker(
    chunk: Metashape.Chunk,
    file_name: str,
) -> None:
    """Write Marker image coordinates to csv file,
    sort by camera, as follows:
    marker1, cam1, x, y
    marker1, cam2, x, y
    ...
    marker1, camN, x, y
    marker2, cam1, x, y
    ....
    markerM, camN, x, y

    Args:
        chunk (Metashape.Chunk): Metashape Chunk
        file_name (str): path of the output csv file
    """

    file = open(file_name, "w")

    for marker in chunk.markers:
        projections = marker.projections  # list of marker projections
        marker_name = marker.label
        for camera in marker.projections.keys():
            x, y = projections[camera].coord
            label = camera.label
            # writing output to file
            file.write(f"{marker_name},{label},{x:.4f},{y:.4f}\n")

    file.close()
    print("Marker exported successfully")


def write_marker_world_coordinates(
    chunk: Metashape.Chunk,
    file_name: str,
) -> None:
    """Write Marker world coordinates to csv file as:
    marker1, X, Y, Z
    ...
    markerM, X, Y, Z

    Args:
        chunk (Metashape.Chunk): Metashape Chunk
        file_name (str): path of the output csv file
    """

    file = open(file_name, "w")

    for marker in chunk.markers:
        marker_name = marker.label
        X, Y, Z = marker.reference.location
        # writing output to file
        file.write(f"{marker_name:5},{X:15.4f},{Y:15.4f},{Z:15.4f}\n")

    file.close()
    print("Marker exported successfully")


if __name__ == "__main__":
    filename = "C:/Users/Francesco/metashape/data/gcps.txt"
    data = read_gcp_file(filename)
    gcps = arrange_gcp(data)
    print(gcps)
