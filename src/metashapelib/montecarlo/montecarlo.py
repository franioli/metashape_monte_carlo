import csv
import math
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Union

import Metashape
import numpy as np
from metashapelib.montecarlo import mc_utils
from metashapelib import utils as ms_utils
from metashapelib import workflow as ms_workflow
from metashapelib.logger import setup_logger

logger = setup_logger(name="MC", log_level="DEBUG")

if not ms_utils.check_license():
    raise Exception(
        "No licence found. Please check that you linked your license (floating or standalone) wih the Metashape python module."
    )
backward_compatibility = ms_utils.backward_compatibility()

NaN = float("NaN")

# NOTE: Currentely, you MUST use psz as the extension for the simulation project files. This is because Metashape do not allow to overwrite the image coordinats of the tie points with the standard .psx extension. This is probably a limitation of the Metashape API.
metashape_simu_ext = "psz"
default_intrinsics_optim = dict = {
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


def initialise_simulation(
    project_path: Path,
    simu_dir: Path = None,
    pts_offset: Metashape.Vector = Metashape.Vector([NaN, NaN, NaN]),
    skip_optimisation: bool = False,
    optimise_intrinsics: Dict = default_intrinsics_optim,
) -> None:
    logger.info("Initializing...")

    # Initialisation
    ref_project_path = Path(project_path)
    if not ref_project_path.exists() or not ref_project_path.suffix == ".psx":
        raise FileNotFoundError(f"File {ref_project_path} does not exist.")
    if not (ref_project_path.parent / (ref_project_path.stem + ".files")).exists():
        raise FileNotFoundError(
            f"File {ref_project_path.parent / (ref_project_path.stem + '.files')} does not exist."
        )

    # Set the simulation directory
    if not simu_dir:
        simu_dir = ref_project_path.parent / "simulation"
    else:
        simu_dir = Path(simu_dir)
    simu_dir.mkdir(parents=True, exist_ok=True)

    project_path = simu_dir / f"simu.{metashape_simu_ext}"
    if project_path.exists():
        project_path.unlink()

    # Open the reference document and copy it to a new document
    ref_doc = Metashape.Document()
    ref_doc.open(str(ref_project_path))
    doc = ref_doc.copy()
    ms_workflow.save_project(doc, path=project_path)

    # Close the reference document
    ref_doc.read_only = False
    del ref_doc

    # Get the chunk from the document
    original_chunk = doc.chunk
    original_chunk.label = "original_chunk"

    # Get the Chunk coordinate system or set it to a local system if it is not set
    if original_chunk.crs is None:
        crs = Metashape.CoordinateSystem(
            'LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'
        )
        original_chunk.crs = crs
    else:
        crs = original_chunk.crs

    # Export a text file with the coordinate system
    with open(simu_dir / "_coordinate_system.txt", "w") as f:
        fwriter = csv.writer(f, dialect="excel-tab", lineterminator="\n")
        fwriter.writerow([crs])
        f.close()

    # If required, calculate the mean point coordinate to use as an offset
    if math.isnan(pts_offset[0]):
        # TODO: test this function with UTM coordinates
        pts_offset = mc_utils.compute_coordinate_offset(original_chunk)

    # Expand region to include all possible points
    ms_workflow.expand_region(original_chunk, 1.5)

    # Carry out an initial bundle adjustment to ensure that everything subsequent has a consistent reference starting point.
    if not skip_optimisation:
        logger.info("Optimising cameras...")
        ms_workflow.optimize_cameras(original_chunk, optimise_intrinsics)

    # Save the sparse point cloud as text file including colour and covariance
    # NOTE: disabled for now
    # logger.info("Exporting sparse point cloud with covariance information...")
    # save_sparse(
    #     original_chunk,
    #     simu_dir / "sparse_pts_reference_cov.csv",
    #     save_color=True,
    #     save_cov=True,
    #     sep=",",
    # )

    # Save the used offset to text file
    with open(simu_dir / "_coordinate_local_origin.txt", "w") as f:
        fwriter = csv.writer(f, dialect="excel-tab", lineterminator="\n")
        fwriter.writerow(pts_offset)
        f.close()

    # Export a text file of observation distances and ground dimensions of pixels from which relative precisions can be calculated
    # NOTE: disabled for now
    # logger.info("Exporting observation distances...")
    # mc_utils.compute_observation_distances(original_chunk, simu_dir)

    # Make a copy of the chunk to use and set it as zero-error
    logger.info("Creating zero error chunk...")
    zero_error_chunk = mc_utils.set_chunk_zero_error(
        chunk=original_chunk,
        new_chunk_label="zero_error_chunk",
        inplace=True,
        optimize_cameras=False,
    )
    logger.info("Done")

    # Export the sparse point cloud
    logger.info("Exporting sparse point cloud...")
    if backward_compatibility:
        zero_error_chunk.exportPoints(
            str(simu_dir / "sparse_pts_reference.ply"),
            source_data=Metashape.DataSource.PointCloudData,
            save_normals=True,
            save_colors=True,
            format=Metashape.PointsFormatPLY,
            crs=crs,
            shift=pts_offset,
        )
    else:
        zero_error_chunk.exportPointCloud(
            str(simu_dir / "sparse_pts_reference.ply"),
            source_data=Metashape.DataSource.TiePointsData,
            save_point_color=True,
            save_point_normal=True,
            format=Metashape.PointCloudFormatPLY,
            crs=crs,
            shift=pts_offset,
        )

    # Save the project
    ms_workflow.save_project(doc)

    # Clean up the output directory and create a new one
    logger.info("Creating and cleaning up output directory...")
    out_dir = Path(simu_dir) / "Monte_Carlo_output"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    # Create a temporary directory to simulation runs
    runs_dir = simu_dir / "runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir, ignore_errors=True)

    # Save the zero-error project as reference
    zero_error_path = runs_dir / f"run_ref.{metashape_simu_ext}"
    zero_error_path.parent.mkdir(parents=True)
    doc.save(str(zero_error_path), [zero_error_chunk])

    logger.info("Initialisation complete.")


def check_initialisation(simu_dir: Path) -> bool:
    if isinstance(simu_dir, str):
        simu_dir = Path(simu_dir)
    elif not isinstance(simu_dir, Path):
        raise TypeError(f"simu_dir must be a string or Path, not {type(simu_dir)}")

    runs_dir = simu_dir / "runs"
    if not simu_dir.exists():
        raise FileNotFoundError(
            f"Simulation directory {simu_dir} does not exist. Unable to resume simulation."
        )
    run_ref_doc = runs_dir / f"run_ref.{metashape_simu_ext}"
    if not run_ref_doc.exists():
        raise FileNotFoundError(
            f"Simulation reference project file {run_ref_doc} does not exist. Unable to resume simulation."
        )

    out_dir = simu_dir / "Monte_Carlo_output"
    if not out_dir.exists():
        raise FileNotFoundError(
            f"Output directory {out_dir} does not exist. Unable to resume simulation."
        )
    return True


def run_iteration(
    iterable: tuple,
    # run_idx: int,
    # runs_dir: Path,
    # out_dir: Path,
    # optimise_intrinsics: Dict = default_intrinsics_optim,
) -> bool:
    """
    Runs a single iteration of the simulation.

    This function duplicates a zero-error chunk, adds Gaussian noise to camera coordinates, marker locations, and observations,
    performs a bundle adjustment, and exports the results.

    Parameters:
    run_idx (int): The index of the current run.
    simu_dir (str or Path): The directory where the simulation data is stored.
    out_dir (str or Path): The directory where the output should be stored.
    zero_error_chunk (Metashape.Chunk): The chunk with zero error to be used as a base for the simulation.
    optimise_intrinsics (bool): Flag to indicate whether to optimize camera intrinsics during bundle adjustment.

    Returns:
    None
    """

    cleanup_run = True

    # Unpack the iterable
    run_idx, simu_dir, optimise_intrinsics, pts_offset = iterable
    pts_offset = Metashape.Vector(pts_offset)

    logger.info(f"Run iteration {run_idx}...")

    # Define directrories
    runs_dir = simu_dir / "runs"
    out_dir = simu_dir / "Monte_Carlo_output"
    ref_doc_path = runs_dir / "run_ref.psz"
    run_doc_path = runs_dir / f"run_{run_idx:04d}.{metashape_simu_ext}"

    # Make a hard copy of the zero-error reference project
    if not ref_doc_path.exists():
        raise FileNotFoundError(f"Reference project {ref_doc_path} does not exist.")
    if run_doc_path.exists():
        logger.debug(
            f"Current iteration project {run_doc_path} already exists. Overwriting..."
        )
        run_doc_path.unlink()
    shutil.copyfile(ref_doc_path, run_doc_path)

    # Read the current project
    if not run_doc_path.exists():
        raise FileNotFoundError(f"File {run_doc_path} does not exist.")
    run_doc = Metashape.Document()
    run_doc.open(str(run_doc_path))
    chunk = run_doc.chunk
    chunk.label = f"simu_chunk_{run_idx:04d}"

    # Add noise to camera coordinates if they are used for georeferencing
    logger.debug(f"Run {run_idx} - Adding Gaussian noise to camera coordinates...")
    mc_utils.add_cameras_gauss_noise(chunk)
    logger.debug(f"Run {run_idx} - Ok.")

    # Add noise to the marker locations
    logger.debug(f"Run {run_idx} - Adding Gaussian noise to marker locations...")
    mc_utils.add_markers_gauss_noise(chunk)
    logger.debug(f"Run {run_idx} - Ok.")

    # Add noise to the observations
    logger.debug(f"Run {run_idx} - Adding Gaussian noise to observations...")
    mc_utils.add_observations_gauss_noise(chunk)
    logger.debug(f"Run {run_idx} - Ok.")

    # Run Bundle adjustment
    logger.debug(f"Run {run_idx} - Optimising cameras...")
    optim = {**default_intrinsics_optim, **optimise_intrinsics}
    chunk.optimizeCameras(
        fit_f=optim["f"],
        fit_cx=optim["cx"],
        fit_cy=optim["cy"],
        fit_b1=optim["b1"],
        fit_b2=optim["b2"],
        fit_k1=optim["k1"],
        fit_k2=optim["k2"],
        fit_k3=optim["k3"],
        fit_k4=optim["k4"],
        fit_p1=optim["p1"],
        fit_p2=optim["p2"],
        tiepoint_covariance=optim["tiepoint_covariance"],
    )
    logger.debug(f"Run {run_idx} - Ok.")

    # Export the results
    try:
        export_results(chunk, run_idx, out_dir, pts_offset)
        logger.info(f"Run {run_idx} - Exported results.")
    except Exception as e:
        logger.error(f"Error exporting results for run {run_idx}: {e}")

    # For debugging purposes, save the project
    # run_doc.save()

    # Clean up the run directory
    if cleanup_run:
        run_doc_path.unlink()

    del chunk
    del run_doc

    logger.info(f"Finished run {run_idx}.")

    ## Old method by copying the reference chunk
    # ref_doc = Metashape.Document()
    # ref_doc.open(str(runs_dir / "run_ref/run.psz"))
    # chunk = ref_doc.chunks[0].copy()
    # chunk.label = f"simu_chunk_{run_idx:04d}"
    # ref_doc.delete(chunk)
    # ref_doc.save()

    return True


def export_results(
    chunk: Metashape.Chunk,
    run_idx: int,
    out_dir: Path,
    pts_offset: Metashape.Vector = Metashape.Vector([NaN, NaN, NaN]),
):
    """
    Exports the results of a processing chunk in various formats.

    This function exports the control point locations, cameras, calibrations, and the sparse point cloud of a processing chunk.
    The exported files are saved in the specified output directory with a unique name constructed from the run index and chunk parameters.

    Parameters:
    chunk (Metashape.Chunk): The processing chunk whose results are to be exported.
    run_idx (int): The index of the current run. Used in constructing the output file names.
    out_dir (Path): The directory where the output files will be saved.

    Returns:
    None
    """
    act_marker_flags = [m.reference.enabled for m in chunk.markers]
    num_act_markers = sum(act_marker_flags)
    crs = chunk.crs

    # Construct the output file names
    basename = f"{run_idx:04d}_MA{chunk.marker_location_accuracy[0]:0.5f}_PA{chunk.marker_projection_accuracy:0.5f}_TA{chunk.tiepoint_accuracy:0.5f}_NAM{num_act_markers:03d}"

    # Export the control point locations
    chunk.exportReference(
        str(out_dir / (basename + "_GC.txt")),
        Metashape.ReferenceFormatCSV,
        items=Metashape.ReferenceItemsMarkers,
        delimiter=",",
        columns="noxyzUVWuvw",
    )
    logger.debug(f"Exported control points to {out_dir / (basename + '_GC.txt')}")

    chunk.exportReference(
        str(out_dir / (basename + "_cams_c.txt")),
        Metashape.ReferenceFormatCSV,
        items=Metashape.ReferenceItemsCameras,
        delimiter=",",
        columns="noxyzabcUVWDEFuvwdef",
    )
    logger.debug(f"Exported cameras to {out_dir / (basename + '_cams_c.txt')}")

    # Export the cameras
    chunk.exportCameras(
        str(out_dir / (basename + "_cams.xml")),
        format=Metashape.CamerasFormatXML,
        crs=crs,
    )  # , rotation_order=Metashape.RotationOrderXYZ)
    logger.debug(f"Exported cameras to {out_dir / (basename + '_cams.xml')}")

    # Export the calibrations
    for sensorIDx, sensor in enumerate(chunk.sensors):
        sensor.calibration.save(
            str(out_dir / (basename + f"_cal{sensorIDx + 1:01d}.xml"))
        )
    logger.debug(f"Exported calibrations to {out_dir / (basename + '_cal*.xml')}")

    # Export the sparse point cloud
    if backward_compatibility:
        chunk.exportPoints(
            str(out_dir / (basename + "_pts.ply")),
            source_data=Metashape.DataSource.PointCloudData,
            save_normals=False,
            save_colors=False,
            format=Metashape.PointsFormatPLY,
            crs=crs,
            shift=pts_offset,
        )
    else:
        chunk.exportPointCloud(
            str(out_dir / (basename + "_pts.ply")),
            source_data=Metashape.DataSource.TiePointsData,
            save_point_normal=False,
            save_point_color=False,
            format=Metashape.PointCloudFormatPLY,
            crs=crs,
            shift=pts_offset,
        )

    logger.debug(f"Exported sparse point cloud to {out_dir / (basename + '_pts.ply')}")


if __name__ == "__main__":
    pass
