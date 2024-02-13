import csv
import math
import os
import random
import threading
from pathlib import Path
from typing import Union

import Metashape

# Reset the random seed, so that all equivalent runs of this script are started identically
random.seed(1)


def save_project(
    doc: Metashape.Chunk, path: Union[str, Path] = None, wait_saved: bool = True
):
    def save(doc, path=None):
        doc.read_only = False
        if path is not None:
            doc.save(str(path))
        else:
            doc.save()

    if doc.path is None and path is None:
        raise ValueError(
            "Document has not been saved yet and no path is specified. Please specify a path to save the document."
        )

    if path is not None:
        x = threading.Thread(target=save, args=(doc, path))
    else:
        x = threading.Thread(target=save, args=(doc,))
    x.start()

    if wait_saved:
        x.join()

    return x


def expand_region(chunk, resize_fct: float) -> None:
    chunk.resetRegion()
    chunk.region.size = resize_fct * chunk.region.size


def duplicate_chunk(
    chunk: Metashape.Chunk,
    new_name: str = None,
) -> Metashape.Chunk:
    new_chunk = chunk.copy()
    if new_name is not None:
        new_chunk.label = new_name
    return new_chunk


def compute_coordinate_offset(chunk: Metashape.Chunk):
    crs = chunk.crs
    if crs is None:
        raise ValueError("Chunk coordinate system is not set")

    points = [point.coord for point in chunk.point_cloud.points if point.valid]

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

    points = chunk.point_cloud.points
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
            for proj in chunk.point_cloud.projections[camera]:
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


def set_chunk_zero_error(chunk: Metashape.Chunk, dir_path: str = None):
    crs = chunk.crs

    # Set the marker locations be zero error, from which we add simulated error
    for marker in chunk.markers:
        if marker.position is not None:
            marker.reference.location = crs.project(
                chunk.transform.matrix.mulp(marker.position)
            )

    # Set the marker and point projections to be zero error, from which we add simulated error
    points = chunk.point_cloud.points
    point_proj = chunk.point_cloud.projections
    npoints = len(points)
    for camera in chunk.cameras:
        if not camera.transform:
            continue

        point_index = 0
        for proj in point_proj[camera]:
            track_id = proj.track_id
            while point_index < npoints and points[point_index].track_id < track_id:
                point_index += 1
            if point_index < npoints and points[point_index].track_id == track_id:
                if not points[point_index].valid:
                    continue
                proj.coord = camera.project(points[point_index].coord)

        # Set the marker points be zero error, from which we can add simulated error
        for markerIDx, marker in enumerate(chunk.markers):
            if (not marker.projections[camera]) or (
                not chunk.markers[markerIDx].position
            ):
                continue
            marker.projections[camera].coord = camera.project(
                chunk.markers[markerIDx].position
            )

    # Export this 'zero error' marker data to file
    if dir_path:
        chunk.exportMarkers(os.path.join(dir_path, "referenceMarkers.xml"))


def add_cameras_gauss_noise(chunk: Metashape.Chunk, sigma: float = None):
    for cam in chunk.cameras:
        # Skip cameras without a transform
        if not cam.transform:
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
        # If no sigma is provided for each camera, use the chunk's marker accuracy
        if not sigma:
            if not marker.reference.accuracy:
                sigma = chunk.marker_location_accuracy
            else:
                sigma = marker.reference.accuracy

        noise = Metashape.Vector([random.gauss(0, s) for s in sigma])
        marker.reference.location += noise


def add_observations_gauss_noise(chunk: Metashape.Chunk):
    tie_proj_x_stdev = chunk.tiepoint_accuracy / math.sqrt(2)
    tie_proj_y_stdev = chunk.tiepoint_accuracy / math.sqrt(2)
    marker_proj_x_stdev = chunk.marker_projection_accuracy / math.sqrt(2)
    marker_proj_y_stdev = chunk.marker_projection_accuracy / math.sqrt(2)
    point_proj = chunk.point_cloud.projections

    for camera in chunk.cameras:
        # Skip cameras without a transform
        if not camera.transform:
            continue

        # Tie points (matches)
        projections = point_proj[camera]
        for proj in projections:
            noise = Metashape.Vector(
                [random.gauss(0, tie_proj_x_stdev), random.gauss(0, tie_proj_y_stdev)]
            )
            proj.coord += noise

        # Markers
        for marker in chunk.markers:
            if not marker.projections[camera]:
                continue
            noise = Metashape.Vector(
                [
                    random.gauss(0, marker_proj_x_stdev),
                    random.gauss(0, marker_proj_y_stdev),
                ]
            )
            marker.projections[camera].coord += noise
