from pathlib import Path

import Metashape
import numpy as np
from metashapelib.utils import backward_compatibility
from metashapelib.workflow import optimize_cameras


def get_ms_tie_points(chunk: Metashape.Chunk):
    backward = backward_compatibility()
    if not backward:
        return chunk.tie_points
    else:
        return chunk.point_cloud


# def compute_reprojection_error_statistics(chunk: Metashape.Chunk, verbose: bool = True):
#     points = chunk.point_cloud.points
#     projections = chunk.point_cloud.projections
#     cam_to_process = [cam for cam in doc.chunk.cameras if cam.transform]

#     res_by_cam = {cam.label: [] for cam in cam_to_process}
#     for camera in cam_to_process:
#         point_index = 0
#         for proj in projections[camera]:
#             track_id = proj.track_id
#             while point_index < len(points) and points[point_index].track_id < track_id:
#                 point_index += 1
#             if point_index < len(points) and points[point_index].track_id == track_id:
#                 if not points[point_index].valid:
#                     continue
#                 dist = camera.error(points[point_index].coord, proj.coord).norm() ** 2
#                 res_by_cam[camera.label].append(dist)
#         if verbose:
#             reprojection_rmse = math.sqrt(
#                 sum(res_by_cam[camera.label]) / len(res_by_cam[camera.label])
#             )
#             print(
#                 f"{camera.label}: {len(res_by_cam[camera.label])} pts, RMSE: {reprojection_rmse:.3f} px"
#             )

#     error = []
#     for k, v in res_by_cam.items():
#         error.extend(v)
#     reprojection_rmse = round(math.sqrt(sum(error) / len(error)), 2)
#     reprojection_max = round(math.sqrt(max(error)), 2)
#     reprojection_std = round(statistics.stdev(error), 2)


# compute_reprojection_error_statistics(chunk)


def compute_reprojection_error(
    chunk: Metashape.Chunk, verbose: bool = False, include_markers: bool = False
) -> dict:
    tie_points = get_ms_tie_points(chunk)
    points = tie_points.points
    projections = tie_points.projections
    cam_to_process = [cam for cam in doc.chunk.cameras if cam.transform]

    res_by_cam = {cam.label: [] for cam in cam_to_process}
    for camera in cam_to_process:
        point_index = 0
        err_list = []
        for proj in projections[camera]:
            track_id = proj.track_id
            while point_index < len(points) and points[point_index].track_id < track_id:
                point_index += 1
            if point_index < len(points) and points[point_index].track_id == track_id:
                if not points[point_index].valid:
                    continue
                err = np.array(
                    camera.error(points[point_index].coord, proj.coord)
                ).reshape(1, 2)
                err_list.append(err)

        if include_markers:
            for marker in chunk.markers:
                proj = marker.projections[camera]

                if not proj:
                    continue
                T = chunk.transform.matrix
                if chunk.crs is not None:
                    coor3d = chunk.crs.unproject(marker.reference.location)
                else:
                    coor3d = marker.reference.location
                coor3d = T.inv().mulp(coor3d)
                err = np.array(camera.error(coor3d, proj.coord)).reshape(1, 2)
                err_list.append(err)
        res_by_cam[camera.label] = np.array(err_list).reshape(-1, 2)

    if verbose:
        for k, v in res_by_cam.items():
            squared_norms = np.linalg.norm(v, axis=1) ** 2
            reprojection_rmse = np.sqrt(squared_norms.mean())
            print(f"{k}: {len(squared_norms)} pts, RMSE: {reprojection_rmse:.3f} px")

    v = np.concatenate([v.ravel() for v in res_by_cam.values()], axis=0)

    return v, res_by_cam


def sigma02(residuals: np.ndarray, n_obs: int, n_params: int, Q=None):
    if Q is None:
        sigma02 = np.dot(residuals, residuals) / (n_obs - n_params)
    else:
        from sksparse.cholmod import cholesky

        # Cholesky decomposition of Q
        Q_chol = cholesky(Q)

        # Compute sigma02
        Q_chol_res = Q_chol @ residuals
        sigma02 = Q_chol_res.T @ Q_chol_res / (n_obs - n_params)

    return sigma02


def compute_sigma02(chunk: Metashape.Chunk, verbose: bool = False):
    # Compute the reprojection error
    v, _ = compute_reprojection_error(chunk, include_markers=True)

    # Compute the number of observations and parameters
    tie_points = get_ms_tie_points(chunk)
    n_obs = len(v)
    n_params = (
        6 * len(chunk.cameras) + 3 * len(tie_points.points) + 7 * len(chunk.sensors)
    )

    # Compute sigma02
    # sigma02 = sigma02(v, n_obs, n_params)
    sigma02 = np.dot(v, v) / (n_obs - n_params)

    if verbose:
        print(f"sigma02 = {sigma02:.5f}")

    return sigma02


# # Prior tie point accuracy of 1
# s_prior = 100
# chunk.tiepoint_accuracy = s_prior
# chunk.marker_projection_accuracy = s_prior
# optimize_cameras(chunk, optimise_intrinsics)

# # Compute the reprojection error
# v, _ = compute_reprojection_error(chunk)

# # Compute the number of observations and parameters
# n_obs = len(v)  # number of observations = 169862
# n_params = (
#     6 * len(chunk.cameras) + 3 * len(chunk.point_cloud.points) + 7 * len(chunk.sensors)
# )  # number of parameters = 51347
# sigma02 = compute_sigma02(v, n_obs, n_params)
# print(f"sigma02 = {sigma02:.5f}")

if __name__ == "__main__":
    # Directory where output will be stored and active control file is saved.
    project_dir = "data/rossia"
    project_name = "rossia_C_bis.psx"

    # Define the camera parameter set to optimise in the bundle adjustment.
    optimise_intrinsics = {
        "f": True,
        "cx": True,
        "cy": True,
        "b1": False,
        "b2": False,
        "k1": True,
        "k2": True,
        "k3": True,
        "k4": False,
        "p1": True,
        "p2": True,
        "tiepoint_covariance": True,
    }

    # Initialisation
    project_dir = Path(project_dir)
    project_path = project_dir / project_name
    doc = Metashape.Document()
    doc.open(str(project_path))
    chunk = doc.chunk

    # Prior tie point accuracy of 0.01
    s_prior = 1
    chunk.tiepoint_accuracy = s_prior
    chunk.marker_projection_accuracy = s_prior
    optimize_cameras(chunk, optimise_intrinsics)
    v, residuals_by_cam = compute_reprojection_error(chunk, include_markers=True)

    # Compute the number of observations and parameters
    tie_points = get_ms_tie_points(chunk)
    n_obs = len(v)
    n_params = (
        6 * len(chunk.cameras) + 3 * len(tie_points.points) + 7 * len(chunk.sensors)
    )

    # Compute sigma02
    s02 = sigma02(v, n_obs, n_params)
    print(f"sigma02 = {sigma02:.5f}")

    # Tests
    marker = chunk.markers[0]
    T = chunk.transform.matrix
    if (
        chunk.transform.translation
        and chunk.transform.rotation
        and chunk.transform.scale
    ):
        T = chunk.crs.localframe(T.mulp(chunk.region.center)) * T
    R = T.rotation() * T.scale()

    cov = marker.position_covariance
    cov = np.array(R * cov * R.t()).reshape(3, 3)

    std = np.sqrt(np.diag(cov))
    std_scaled = np.sqrt(s02 * np.diag(cov))

    print("done.")
