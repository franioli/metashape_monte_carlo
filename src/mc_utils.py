import csv
import math
import os
import random

import Metashape
import numpy as np

from src.utils import backward_compatibility

# Reset the random seed, so that all equivalent runs of this script are started identically
# random.seed(1)


rng = np.random.default_rng()


def vector_to_cov(cov_vector):
    if len(cov_vector) != 6:
        raise ValueError(
            "Input vector must contain 6 elements: [Var(X), Var(Y), Var(Z), Cov(X,Y), Cov(X,Z), Cov(Y,Z)]"
        )

    var_x, var_y, var_z = cov_vector[:3]
    cov_xy, cov_xz, cov_yz = cov_vector[3:]

    if var_x < 0 or var_y < 0 or var_z < 0:
        raise ValueError("Variances must be non-negative.")

    cov_matrix = np.array(
        [
            [var_x, cov_xy, cov_xz],
            [cov_xy, var_y, cov_yz],
            [cov_xz, cov_yz, var_z],
        ]
    )
    return cov_matrix


def get_ms_tie_points(chunk: Metashape.Chunk):
    backward = backward_compatibility()
    if not backward:
        return chunk.tie_points
    else:
        return chunk.point_cloud


def compute_coordinate_offset(chunk: Metashape.Chunk) -> Metashape.Vector:
    """
    Compute the coordinate offset for a given Metashape chunk.

    Parameters:
        chunk (Metashape.Chunk): The Metashape chunk for which to compute the coordinate offset.

    Returns:
        Metashape.Vector: The computed coordinate offset as a Metashape Vector.

    Raises:
        ValueError: If the chunk coordinate system is not set.

    Notes:
        - The coordinate offset is computed by averaging the coordinates of all valid tie points in the chunk.
        - If the chunk coordinate system is not set, a ValueError is raised.

    """
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

    Parameters:
        chunk (Metashape.Chunk): The chunk containing the tie points and cameras.
        dir_path (str): The directory path where the text file will be saved.

    Returns:
        str: The file path of the exported text file.

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
    """
    Set the chunk's error to zero by adjusting marker and camera locations.

    Parameters:
        chunk (Metashape.Chunk): The chunk to modify.
        optimize_cameras (bool, optional): Whether to optimize camera parameters after setting error to zero. Defaults to False.
        otimize_params (dict, optional): Optimization parameters for camera optimization. Defaults to {}.
        inplace (bool, optional): Whether to modify the chunk in place or create a copy. Defaults to False.
        new_chunk_label (str, optional): The label for the new chunk if not modifying in place. Defaults to None.

    Returns:
        Metashape.Chunk: The modified chunk with error set to zero.

    Note:
        This function assumes that the chunk has tie points and projections available.

    """
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
    """
    Add Gaussian noise to the location of cameras in a Metashape chunk.

    Parameters:
        chunk (Metashape.Chunk): The chunk containing the cameras.
        sigma (float, optional): The standard deviation of the Gaussian noise. If not provided, the camera accuracy from the chunk or the camera's reference data will be used.

    Returns:
        None

    Notes:
        - Cameras without a transform or without reference data enabled will be skipped.
        - The noise vector is computed using random.gauss(0, sigma) for each dimension of the camera location.
        - The computed noise vector is added to the camera location.

    Example:
        >>> chunk = Metashape.app.document.chunk
        >>> add_cameras_gauss_noise(chunk, sigma=0.1)
    """

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


def add_markers_gauss_noise(
    chunk: Metashape.Chunk,
    std: float = None,
    cov: dict = None,
):
    """
    Add Gaussian noise to the locations of active markers in a Metashape chunk.

    Parameters:
        chunk (Metashape.Chunk): The chunk containing the markers.
        std (float, optional): The standard deviation of the Gaussian noise. If a scalar value is provided, the same standard deviation is used for all three dimensions. If a 3-element vector is provided, each dimension can have a different standard deviation. Defaults to None.
        cov (dict, optional): A dictionary containing the covariance matrix for each marker. The keys of the dictionary must be the marker labels, and the values must be either a 3x3 covariance matrix or a 6-element vector containing the variances and covariances. Defaults to None.

    Raises:
        ValueError: If sigma is not a scalar or a 3-element vector, or if cov is not a dictionary, or if the covariance matrix for a marker is not provided or is not a valid shape.

    Returns:
        None

    Note:
        - If both sigma and cov are None, the noise vector for each marker is computed using the chunk's marker accuracy.
        - The noise is added to the location of each active marker in the chunk.
        - Check points are not affected by the noise.

    Example usage:
        >>> chunk = Metashape.app.document.chunk
        >>> sigma = 0.1
        >>> cov = {"Marker1": np.eye(3), "Marker2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
        >>> add_markers_gauss_noise(chunk, sigma, cov)
    """
    num_active_markers = len(
        [marker for marker in chunk.markers if marker.reference.enabled]
    )

    if std is not None:
        if isinstance(std, (int, float)):
            std = [std, std, std]
        elif len(std) != 3:
            raise ValueError("Sigma must be a scalar or a 3-element vector.")
        noise = rng.normal(loc=0, scale=std, size=(num_active_markers, 3))

    if cov is not None:
        if not isinstance(cov, dict):
            raise ValueError(
                "Covariance must be a dictionary containing the covariance matrix for each marker. The keys must be the marker label and the values must be a 3x3 matrix or a 6-element vector."
            )
        noise = []
        marker_labels = [
            marker.label for marker in chunk.markers if marker.reference.enabled
        ]
        for label in marker_labels:
            if label not in cov:
                raise ValueError(f"No covariance matrix provided for marker {label}")
            else:
                cov_mat = cov[label]
                if cov_mat.shape == (6,):
                    cov_mat = vector_to_cov(cov)
                elif cov_mat.shape != (3, 3):
                    raise ValueError(
                        "Covariance matrix must be 3x3 or a 6-element vector containing the threw variances and three covariances."
                    )
                noise.append(rng.multivariate_normal(mean=np.zeros(3), cov=cov_mat))

        noise = np.array(noise)

    for i, marker in enumerate(chunk.markers):
        # Do not add noise to check points
        if not marker.reference.enabled:
            continue

        # If no sigma or covariance matrix is provided for each camera, use the chunk's marker accuracy
        if std is None and cov is None:
            # If no sigma is provided for each marker, use the chunk's marker accuracy
            if not marker.reference.accuracy:
                sigma = chunk.marker_location_accuracy
            else:
                sigma = marker.reference.accuracy

            # Compute the noise vector for this marker
            noise = Metashape.Vector([random.gauss(0, s) for s in sigma])
        else:
            # Use the precomputed noise vector
            noise = Metashape.Vector(noise[i])

        marker.reference.location += noise


def add_observations_gauss_noise(chunk: Metashape.Chunk):
    """
    Add Gaussian noise to the observations in a Metashape chunk.

    Parameters:
        chunk (Metashape.Chunk): The chunk containing the observations.

    Returns:
        None

    Raises:
        None

    Notes:
        - This function adds Gaussian noise to the tie point projections and marker projections in a Metashape chunk.
        - The standard deviation of the noise is calculated based on the tie point accuracy and marker projection accuracy of the chunk.
        - The noise is generated using numpy's random number generator for faster computation.

    Example:
        add_observations_gauss_noise(chunk)
    """
    tie_proj_stdev = chunk.tiepoint_accuracy / math.sqrt(2)
    marker_proj_stdev = chunk.marker_projection_accuracy / math.sqrt(2)

    # Get tie points (projections of 3D points on images) for all images
    tie_points = get_ms_tie_points(chunk)
    point_proj = tie_points.projections

    for camera in chunk.cameras:
        # Skip cameras without estimated exterior orientation
        if not camera.transform:
            continue

        # Get the projections for the current image
        projections = point_proj[camera]
        n_points = len(projections)

        # Generate noise for all the projections using numpy (faster)
        noise = rng.normal(0, tie_proj_stdev, (n_points, 2))

        # Add noise to the projections
        for proj, n in zip(projections, noise):
            # n = [
            #     random.gauss(0, tie_proj_stdev),
            #     random.gauss(0, tie_proj_stdev),
            # ]
            n = Metashape.Vector(n)
            proj.coord += n

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
