from typing import Union

import Metashape
import numpy as np

""" License and version"""


def check_license() -> bool:
    if Metashape.app.activated:
        print("Metashape is activated: ", Metashape.app.activated)
        return True
    else:
        print(
            "No licence found. Please check that you linked your license (floating or standalone) wih the Metashape python module."
        )
        return False


def get_version() -> str:
    return Metashape.app.version


def backward_compatibility() -> bool:
    if get_version() < "2.0":
        return True
    else:
        return False


""" Get objects"""


def get_marker_by_label(chunk, label):
    for marker in chunk.markers:
        if marker.label == label:
            return marker
    return None


def get_camera_by_label(chunk, label):
    for camera in chunk.cameras:
        if camera.label.lower() == label.lower():
            return camera
    return None


def get_sensor_id_by_label(
    chunk: Metashape.Chunk,
    sensor_label: str,
) -> int:
    sensors = chunk.sensors
    for s_id in sensors:
        sensor = sensors[s_id]
        if sensor.label == sensor_label:
            return s_id


"""Markers"""


def add_markers(
    chunk: Metashape.Chunk,
    X: np.ndarray,
    projections: dict,
    label: str = None,
    accuracy: Union[float, np.ndarray] = None,
) -> None:
    # Create Markers given its 3D object coordinates
    X = Metashape.Vector(X)
    X_ = chunk.transform.matrix.inv().mulp(X)
    marker = chunk.addMarker(X_)

    # Add projections on images given image coordinates in a  dictionary, as  {im_name: (x,y)}
    for k, v in projections.items():
        cam = get_camera_by_label(chunk, k.split(".")[0])
        marker.projections[cam] = Metashape.Marker.Projection(Metashape.Vector(v))

    # If provided, add label and a-priori accuracy
    if label:
        marker.label = label
    if accuracy:
        marker.reference.accuracy = accuracy
    marker.enabled = True
    marker.reference.enabled = True


""" MISCELLANEOUS """


def make_homogeneous(
    v: Metashape.Vector,
) -> Metashape.Vector:
    vh = Metashape.Vector([1.0 for x in range(v.size + 1)])
    for i, x in enumerate(v):
        vh[i] = x

    return vh


def make_inomogenous(
    vh: Metashape.Vector,
) -> Metashape.Vector:
    v = vh / vh[vh.size - 1]
    return v[: v.size - 1]
