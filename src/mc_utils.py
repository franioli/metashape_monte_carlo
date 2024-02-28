import csv
import math
import os
import random

import Metashape

from src.utils import backward_compatibility

# # Reset the random seed, so that all equivalent runs of this script are started identically
# random.seed(1)


def get_ms_tie_points(chunk: Metashape.Chunk):
    backward = backward_compatibility()
    if not backward:
        return chunk.tie_points
    else:
        return chunk.point_cloud


def compute_coordinate_offset(chunk: Metashape.Chunk):
    crs = chunk.crs
    if crs is None:
        raise ValueError("Chunk coordinate system is not set")

    tie_points = get_ms_tie_points(chunk)
    points = [point.coord for point in tie_points.points if point.valid]

    if not points:
        return Metashape.Vector([0, 0, 0])

    npoints = len(points)
    pts_offset = Metashape.Vector(map(sum, zip(*points))) / npoints

    pts_offset = crs.project(chunk.transform.matrix.mulp(pts_offset[:3]))
    pts_offset = Metashape.Vector(round(coord, -2) for coord in pts_offset)

    return pts_offset


def compute_observation_distances(chunk: Metashape.Chunk, dir_path: str) -> str:
    """
    Export a text file of observation distances and ground dimensions of pixels from which relative precisions can be calculated. The File will have one row for each observation, and three columns:
        cameraID	  ground pixel dimension (m)   observation distance (m)

    """

    tie_points = get_ms_tie_points(chunk)
    points = tie_points.points
    npoints = len(points)
    camera_index = 0
    fpath = os.path.join(dir_path, "_observation_distances.txt")
    with open(fpath, "w") as f:
        fwriter = csv.writer(f, dialect="excel-tab", lineterminator="\n")
        for camera in chunk.cameras:
            camera_index += 1
            if not camera.transform:
                continue

            fx = camera.sensor.calibration.f

            point_index = 0
            for proj in tie_points.projections[camera]:
                track_id = proj.track_id
                while point_index < npoints and points[point_index].track_id < track_id:
                    point_index += 1
                if point_index < npoints and points[point_index].track_id == track_id:
                    if not points[point_index].valid:
                        continue
                    dist = (
                        chunk.transform.matrix.mulp(camera.center)
                        - chunk.transform.matrix.mulp(
                            Metashape.Vector(
                                [
                                    points[point_index].coord[0],
                                    points[point_index].coord[1],
                                    points[point_index].coord[2],
                                ]
                            )
                        )
                    ).norm()
                    fwriter.writerow(
                        [
                            camera_index,
                            "{0:.4f}".format(dist / fx),
                            "{0:.2f}".format(dist),
                        ]
                    )

        f.close()

    return fpath


def set_chunk_zero_error(
    chunk: Metashape.Chunk,
    optimize_cameras: bool = False,
    otimize_params: dict = {},
    inplace: bool = False,
    new_chunk_label: str = None,
):
    # Make a copy of the chunk if not inplace
    if not inplace:
        chunk_ = chunk.copy()

        # Set the chunk label
        if new_chunk_label:
            chunk_.label = new_chunk_label
    else:
        chunk_ = chunk

    # Get the chunk_ coordinate system
    crs = chunk_.crs

    # Set the sensor estimated internal parameters to the reference
    for sensor in chunk_.sensors:
        if sensor.user_calib:
            sensor.user_calib = sensor.calibration

    # Set the marker locations be zero error, if the marker is enabled (GCPs)
    for marker in chunk_.markers:
        if marker.position is not None and marker.reference.enabled:
            marker.reference.location = crs.project(
                chunk_.transform.matrix.mulp(marker.position)
            )

    # Set the camera locations be zero error, if the camera has prior locations
    for camera in chunk_.cameras:
        if camera.transform and camera.reference.enabled:
            camera.reference.location = crs.project(
                chunk_.transform.matrix.mulp(camera.center)
            )

    # Set the marker and point projections to be zero error, from which we add simulated error
    tie_points = get_ms_tie_points(chunk)
    points = tie_points.points
    point_proj = tie_points.projections
    npoints = len(points)
    for camera in chunk_.cameras:
        if not camera.transform:
            continue

        # Set the point projections be zero error
        point_index = 0
        for proj in point_proj[camera]:
            track_id = proj.track_id
            while point_index < npoints and points[point_index].track_id < track_id:
                point_index += 1
            if point_index < npoints and points[point_index].track_id == track_id:
                if not points[point_index].valid:
                    continue
                proj.coord = camera.project(points[point_index].coord)

        # Set the marker projections be zero error
        for marker in chunk_.markers:
            if (not marker.projections[camera]) or (not marker.position):
                continue
            marker.projections[camera].coord = camera.project(marker.position)

    if optimize_cameras:
        chunk_.optimizeCameras(
            fit_f=otimize_params.get("f", False),
            fit_cx=otimize_params.get("cx", False),
            fit_cy=otimize_params.get("cy", False),
            fit_b1=otimize_params.get("b1", False),
            fit_b2=otimize_params.get("b2", False),
            fit_k1=otimize_params.get("k1", False),
            fit_k2=otimize_params.get("k2", False),
            fit_k3=otimize_params.get("k3", False),
            fit_k4=otimize_params.get("k4", False),
            fit_p1=otimize_params.get("p1", False),
            fit_p2=otimize_params.get("p2", False),
            tiepoint_covariance=otimize_params.get("tiepoint_covariance", False),
        )

    return chunk_


def add_cameras_gauss_noise(chunk: Metashape.Chunk, sigma: float = None):
    for cam in chunk.cameras:
        # Skip cameras without a transform
        if not cam.transform:
            continue

        # Skip cameras without a reference data enabled
        if not cam.reference.enabled:
            continue

        # If no sigma is provided for each camera, use the chunk's camera accuracy
        if not sigma:
            if not cam.reference.accuracy:
                sigma = chunk.camera_location_accuracy
            else:
                sigma = cam.reference.accuracy

        # Compute the noise vector
        noise = Metashape.Vector([random.gauss(0, s) for s in sigma])

        # Add the noise to the camera location
        cam.reference.location += noise


def add_markers_gauss_noise(chunk: Metashape.Chunk, sigma: float = None):
    for marker in chunk.markers:
        # Do not add noise to check points
        if not marker.reference.enabled:
            continue

        # If no sigma is provided for each camera, use the chunk's marker accuracy
        if not sigma:
            if not marker.reference.accuracy:
                sigma = chunk.marker_location_accuracy
            else:
                sigma = marker.reference.accuracy

        noise = Metashape.Vector([random.gauss(0, s) for s in sigma])
        marker.reference.location += noise


def add_observations_gauss_noise(chunk: Metashape.Chunk):
    tie_proj_stdev = chunk.tiepoint_accuracy / math.sqrt(2)
    marker_proj_stdev = chunk.marker_projection_accuracy / math.sqrt(2)

    tie_points = get_ms_tie_points(chunk)
    point_proj = tie_points.projections

    for camera in chunk.cameras:
        # Skip cameras without estimated exterior orientation
        if not camera.transform:
            continue

        # Tie points (matches)
        projections = point_proj[camera]
        for proj in projections:
            noise = Metashape.Vector(
                [
                    random.gauss(0, tie_proj_stdev),
                    random.gauss(0, tie_proj_stdev),
                ]
            )
            proj.coord += noise

        # Markers
        for marker in chunk.markers:
            if not marker.projections[camera]:
                continue
            noise = Metashape.Vector(
                [
                    random.gauss(0, marker_proj_stdev),
                    random.gauss(0, marker_proj_stdev),
                ]
            )
            if backward_compatibility():
                marker.projections[camera].coord += noise
            else:
                marker.projections[camera].coord.x += noise[0]
                marker.projections[camera].coord.y += noise[1]
