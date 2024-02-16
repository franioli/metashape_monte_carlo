import csv
import math
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Dict

import Metashape

import mc_results
from logger import setup_logger
from src import mc_utils
from src import workflow as ms
from src.export_to_file import save_sparse

NaN = float("NaN")
logger = setup_logger(name="MC", log_level="DEBUG")

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


def export_results(chunk: Metashape.Chunk, run_idx: int, out_dir: Path):
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
    chunk.exportPoints(
        str(out_dir / (basename + "_pts.ply")),
        source_data=Metashape.DataSource.PointCloudData,
        save_normals=False,
        save_colors=False,
        format=Metashape.PointsFormatPLY,
        crs=crs,
        shift=pts_offset,
    )
    logger.debug(f"Exported sparse point cloud to {out_dir / (basename + '_pts.ply')}")


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

    run_idx, runs_dir, out_dir, optimise_intrinsics = iterable

    logger.info(f"Run iteration {run_idx}/ {num_randomisations-1}...")

    # Make a hard copy if the zero-error project
    run_dir = runs_dir / f"run_{run_idx:04d}"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
        logger.debug(f"Directory {run_dir} already exists. Overwriting...")
    shutil.copytree(runs_dir / "run_ref", run_dir)
    logger.info(f"Created directory {run_dir}.")

    # Read the zero-error chunk
    run_doc = Metashape.Document()
    run_doc.open(str(run_dir / "run.psx"))
    chunk = run_doc.chunk

    # Copy the zero-error chunk to use for this simulation
    # chunk = ms.duplicate_chunk(zero_error_chunk, f"simu_chunk_{run_idx:04d}")

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
        export_results(chunk, run_idx, out_dir)
        logger.info(f"Run {run_idx} - Exported results.")
    except Exception as e:
        logger.error(f"Error exporting results for run {run_idx}: {e}")

    # export_thread = threading.Thread(
    #     target=export_results,
    #     args=(run_doc, run_idx),
    # )
    # export_thread.start()
    # export_thread.join()

    # Clean up the run directory
    del chunk
    del run_doc
    shutil.rmtree(run_dir, ignore_errors=True)

    logger.info(f"Finished run {run_idx} / {num_randomisations-1}.")

    return True


def initialise_simulation(
    project_path: Path,
    simu_dir: Path = None,
    optimise_intrinsics: Dict = default_intrinsics_optim,
    pts_offset: Metashape.Vector = Metashape.Vector([NaN, NaN, NaN]),
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

    project_path = simu_dir / "simu.psx"
    if project_path.exists():
        project_path.unlink()
        shutil.rmtree(
            project_path.parent / (project_path.stem + ".files"), ignore_errors=True
        )

    # Open the reference document and copy it to a new document
    ref_doc = Metashape.Document()
    ref_doc.open(str(ref_project_path))
    doc = ref_doc.copy()
    ms.save_project(doc, path=project_path)

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

    # # Find which markers are enabled for use as control points in the bundle adjustment
    # act_marker_flags = [m.reference.enabled for m in original_chunk.markers]
    # num_act_markers = sum(act_marker_flags)

    # # Find which camera orientations are enabled for use as control in the bundle adjustment
    # act_cam_orient_flags = [cam.reference.enabled for cam in original_chunk.cameras]
    # num_act_cam_orients = sum(act_cam_orient_flags)

    # # Derive x and y components for image measurement precisions
    # tie_proj_x_stdev = original_chunk.tiepoint_accuracy / math.sqrt(2)
    # tie_proj_y_stdev = original_chunk.tiepoint_accuracy / math.sqrt(2)
    # marker_proj_x_stdev = original_chunk.marker_projection_accuracy / math.sqrt(2)
    # marker_proj_y_stdev = original_chunk.marker_projection_accuracy / math.sqrt(2)

    # Expand region to include all possible points
    ms.expand_region(original_chunk, 1.5)

    # Carry out an initial bundle adjustment to ensure that everything subsequent has a consistent reference starting point.
    ms.optimize_cameras(original_chunk, optimise_intrinsics)

    # Save the sparse point cloud as text file including colour and covariance
    save_sparse(
        original_chunk,
        simu_dir / "sparse_pts_reference_cov.csv",
        save_color=True,
        save_cov=True,
        sep=",",
    )

    # Save the used offset to text file
    with open(simu_dir / "_coordinate_local_origin.txt", "w") as f:
        fwriter = csv.writer(f, dialect="excel-tab", lineterminator="\n")
        fwriter.writerow(pts_offset)
        f.close()

    # Export a text file of observation distances and ground dimensions of pixels from which relative precisions can be calculated
    # File will have one row for each observation, and three columns:
    # cameraID	  ground pixel dimension (m)   observation distance (m)
    mc_utils.compute_observation_distances(original_chunk, simu_dir)

    # Make a copy of the chunk to use and set it as zero-error
    zero_error_chunk = ms.duplicate_chunk(original_chunk, "zero_error_chunk")
    mc_utils.set_chunk_zero_error(zero_error_chunk)

    # Run Bundle Adjustment with zero-error chunk
    ms.optimize_cameras(zero_error_chunk, optimise_intrinsics)

    # Export the sparse point cloud
    zero_error_chunk.exportPoints(
        str(simu_dir / "sparse_pts_reference.ply"),
        source_data=Metashape.DataSource.PointCloudData,
        save_normals=True,
        save_colors=True,
        format=Metashape.PointsFormatPLY,
        crs=crs,
        shift=pts_offset,
    )

    # Save the project
    ms.save_project(doc)

    # Clean up the output directory and create a new one
    out_dir = Path(simu_dir) / "Monte_Carlo_output"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    # Create a temporary directory to simulation runs
    runs_dir = simu_dir / "runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir, ignore_errors=True)
    zero_error_path = runs_dir / "run_ref" / "run.psx"
    zero_error_path.parent.mkdir(parents=True)
    doc.save(str(zero_error_path), [zero_error_chunk])


def check_initialisation(simu_dir: Path) -> bool:
    if isinstance(simu_dir, str):
        simu_dir = Path(simu_dir)
    elif not isinstance(simu_dir, Path):
        raise TypeError(f"simu_dir must be a string or Path, not {type(simu_dir)}")
    runs_dir = simu_dir / "runs"
    out_dir = simu_dir / "Monte_Carlo_output"
    if not simu_dir.exists():
        raise FileNotFoundError(
            f"Simulation directory {simu_dir} does not exist. Unable to resume simulation."
        )
    if not (simu_dir / "simu.psx").exists():
        raise FileNotFoundError(
            f"Simulation project file {simu_dir / 'simu.psx'} does not exist. Unable to resume simulation."
        )
    if not (runs_dir / "run_ref").exists():
        raise FileNotFoundError(
            f"Simulation reference directory {simu_dir / 'run_ref'} does not exist. Unable to resume simulation."
        )
    if not (runs_dir / "run_ref" / "run.psx").exists():
        raise FileNotFoundError(
            f"Simulation reference project file {simu_dir / 'run_ref' / 'run.psx'} does not exist. Unable to resume simulation."
        )
    if not out_dir.exists():
        raise FileNotFoundError(
            f"Output directory {out_dir} does not exist. Unable to resume simulation."
        )
    return True


def montecarlo_simulation(
    project_path: Path,
    num_randomisations: int,
    simu_dir: Path = None,
    run_parallel: bool = False,
    workers: int = 10,
    resume_sumulations_from: int = -1,
    optimise_intrinsics: Dict = default_intrinsics_optim,
    pts_offset: Metashape.Vector = Metashape.Vector([NaN, NaN, NaN]),
):
    """
    Conducts a Monte Carlo simulation on a given project.

    This function opens a reference project, copies it, and then performs a series of operations on the copy, including bundle adjustment, exporting sparse point clouds, and adding Gaussian noise. The simulation can be run in parallel or sequentially, and the results are saved in a specified directory.

    Args:
        dir_path (str): The directory path where the project is located and where the results will be saved.
        project_name (str): The name of the project on which the simulation will be conducted.
        num_randomisations (int): The number of randomisations to be performed in the simulation.
        run_parallel (bool, optional): Whether to run the simulation in parallel. Defaults to False.
        workers (int, optional): The number of parallel processes to use if run_parallel is True. Defaults to 10.
        optimise_intrinsics (Dict, optional): The parameters for camera optimization. Defaults to default_intrinsics_optim.
        pts_offset (Metashape.Vector, optional): The offset for point coordinates. Defaults to Metashape.Vector([NaN, NaN, NaN]).

    Returns:
        None
    """
    # If the resume_sumulations_from counter is negative, initialise the simulation. If it is positive, skip it and resume the simulation from the specified run number.
    if resume_sumulations_from < 0:
        initialise_simulation(
            project_path=project_path,
            simu_dir=simu_dir,
            optimise_intrinsics=optimise_intrinsics,
            pts_offset=pts_offset,
        )

    # Check the initialisation of the simulation
    if not check_initialisation(simu_dir):
        raise RuntimeError("Cannot resume simulation. Initialisation failed.")
    runs_dir = simu_dir / "runs"
    out_dir = simu_dir / "Monte_Carlo_output"
    logger.info(
        f"Resuming Monte Carlo simulation from run {resume_sumulations_from}..."
    )

    # Reset the resume_sumulations_from counter to zero if it is negative
    if resume_sumulations_from < 0:
        resume_sumulations_from = 0

    # Create the iterable with the parameters of each run
    randomisation = range(resume_sumulations_from, num_randomisations)
    act_num_randomisations = num_randomisations - resume_sumulations_from
    iterable_parms = zip(
        randomisation,
        [runs_dir] * act_num_randomisations,
        [out_dir] * act_num_randomisations,
        [optimise_intrinsics] * act_num_randomisations,
    )

    # Run the simulation
    logger.info("Starting Monte Carlo simulation...")
    if run_parallel:
        with Pool(workers) as pool:
            result = pool.map(
                run_iteration,
                iterable_parms,
            )
            logger.info("Succeded iterations:")
            for value in result:
                print("*", end=" ")
                if not value:
                    print("_", end=" ")
            print("")

        # def callback(result_iterable):
        #     for result in result_iterable:
        #         print(f"Got result: {result}")

        # def error_callback(error):
        #     print(f"Got error: {error}")

        # with Pool(workers) as pool:
        #     result = pool.starmap_async(
        #         run_iteration,
        #         items,
        #         # chunksize=10,
        #         callback=callback,
        #         error_callback=error_callback,
        #     )
        #     for value in result.get():
        #         if not value:
        #             logger.error("Error in parallel processing")
    else:
        result = list(map(run_iteration, iterable_parms))

    logger.info("Simulation completed.")

    return result


if __name__ == "__main__":
    # Directory where output will be stored and active control file is saved.
    # The files will be generated in a sub-folder named "Monte_Carlo_output"
    project_path = "data/rossia/rossia_1gcp.psx"
    # project_path = "data/belvedere2023/belvedere2023.psx"

    project_path = Path(project_path)
    simu_dir = project_path.parent / f"simulation_{project_path.stem}"

    # Define how many times bundle adjustment (Metashape 'optimisation') will be carried out.
    num_randomisations = 1000
    run_parallel = True
    workers = 5
    # NOTE: Keep the number of workers low if running on a single CPU as the Metashape bundle adjustment is already multi-threaded and will use all available cores. If this is set too high, it will slow down the simulation and some runs may stuck.

    # Resume the Monte Carlo simulation from a specific run number. Set to -1 to start a new simulation.
    resume_sumulations_from = -1

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

    # The offset will be subtracted from point coordinates.
    # e.g.  pts_offset = Metashape.Vector( [266000, 4702000, 0] )
    # pts_offset = Metashape.Vector([NaN, NaN, NaN])
    pts_offset = Metashape.Vector([0.0, 0.0, 0.0])

    # Run the Monte Carlo simulation
    montecarlo_simulation(
        project_path=project_path,
        num_randomisations=num_randomisations,
        simu_dir=simu_dir,
        run_parallel=run_parallel,
        workers=workers,
        resume_sumulations_from=resume_sumulations_from,
        optimise_intrinsics=optimise_intrinsics,
        pts_offset=pts_offset,
    )

    # Read and process the output files
    mc_results.main(
        proj_dir=simu_dir,
        pcd_ext="ply",
    )
