import csv
import math
import shutil
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import Metashape

import mc_results
from src import mc_utils
from src import workflow as ms
from src.export_to_file import save_sparse

NaN = float("NaN")


# Function to export the results of a simulation run
def export_results(chunk, run_idx, out_dir):
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
    chunk.exportReference(
        str(out_dir / (basename + "_cams_c.txt")),
        Metashape.ReferenceFormatCSV,
        items=Metashape.ReferenceItemsCameras,
        delimiter=",",
        columns="noxyzabcUVWDEFuvwdef",
    )

    # Export the cameras
    chunk.exportCameras(
        str(out_dir / (basename + "_cams.xml")),
        format=Metashape.CamerasFormatXML,
        crs=crs,
    )  # , rotation_order=Metashape.RotationOrderXYZ)

    # Export the calibrations
    for sensorIDx, sensor in enumerate(chunk.sensors):
        sensor.calibration.save(
            str(out_dir / (basename + f"_cal{sensorIDx + 1:01d}.xml"))
        )

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


# Run the Monte Carlo simulations
def run_iteration(
    run_idx,
    simu_dir,
    out_dir,
    zero_error_chunk,
    optimise_intrinsics,
    num_act_cam_orients,
):
    # print(f"Run {run_idx}/ {num_randomisations}...")

    # # Make a hard copy
    # run_dir = simu_dir / f"run_{run_idx:04d}"
    # shutil.copytree(simu_dir / "run_ref", run_dir)

    # # Read the zero-error chunk
    # run_doc = Metashape.Document()
    # run_doc.open(str(run_dir / "run.psx"))
    # chunk = run_doc.chunk

    # Copy the zero-error chunk to use for this simulation
    chunk = ms.duplicate_chunk(zero_error_chunk, f"simu_chunk_{run_idx:04d}")

    # Add noise to camera coordinates if they are used for georeferencing
    if num_act_cam_orients > 0:
        mc_utils.add_cameras_gauss_noise(chunk)

    # Add noise to the marker locations
    mc_utils.add_markers_gauss_noise(chunk)

    # Add noise to the observations
    mc_utils.add_observations_gauss_noise(chunk)

    # Run Bundle adjustment
    ms.optimize_cameras(chunk, optimise_intrinsics)

    # Export the results
    export_results(chunk, run_idx, out_dir)
    # export_thread = threading.Thread(
    #     target=export_results,
    #     args=(run_doc, run_idx),
    # )
    # export_thread.start()
    # export_thread.join()

    # Clean up
    del chunk
    # del run_doc
    # shutil.rmtree(run_dir, ignore_errors=True)

    # print(f"Finished run {run_idx} / {num_randomisations}.")


def montecarlo_simulation(
    dir_path: str,
    project_name: str,
    num_randomisations: int,
    run_parallel: bool = False,
    parallell_processes: int = 10,
    optimise_intrinsics: dict = {
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
    },
    pts_offset: Metashape.Vector = Metashape.Vector([0.0, 0.0, 0.0]),
):
    # Initialisation
    dir_path = Path(dir_path)
    ref_project_path = dir_path / project_name
    project_path = dir_path / "simu.psx"
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
    with open(dir_path / "_coordinate_system.txt", "w") as f:
        fwriter = csv.writer(f, dialect="excel-tab", lineterminator="\n")
        fwriter.writerow([crs])
        f.close()

    # If required, calculate the mean point coordinate to use as an offset
    if math.isnan(pts_offset[0]):
        # TODO: test this function with UTM coordinates
        pts_offset = mc_utils.compute_coordinate_offset(original_chunk)

    # Find which markers are enabled for use as control points in the bundle adjustment
    act_marker_flags = [m.reference.enabled for m in original_chunk.markers]
    num_act_markers = sum(act_marker_flags)

    # Find which camera orientations are enabled for use as control in the bundle adjustment
    act_cam_orient_flags = [cam.reference.enabled for cam in original_chunk.cameras]
    num_act_cam_orients = sum(act_cam_orient_flags)

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
        dir_path / "sparse_pts_reference_cov.csv",
        save_color=True,
        save_cov=True,
        sep=",",
    )

    # Save the used offset to text file
    with open(dir_path / "_coordinate_local_origin.txt", "w") as f:
        fwriter = csv.writer(f, dialect="excel-tab", lineterminator="\n")
        fwriter.writerow(pts_offset)
        f.close()

    # Export a text file of observation distances and ground dimensions of pixels from which relative precisions can be calculated
    # File will have one row for each observation, and three columns:
    # cameraID	  ground pixel dimension (m)   observation distance (m)
    mc_utils.compute_observation_distances(original_chunk, dir_path)

    # Make a copy of the chunk to use and set it as zero-error
    zero_error_chunk = ms.duplicate_chunk(original_chunk, "zero_error_chunk")
    mc_utils.set_chunk_zero_error(zero_error_chunk, dir_path)

    # Run Bundle Adjustment with zero-error chunk
    ms.optimize_cameras(zero_error_chunk, optimise_intrinsics)

    # Export the sparse point cloud
    zero_error_chunk.exportPoints(
        str(dir_path / "sparse_pts_reference.ply"),
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
    out_dir = Path(dir_path) / "Monte_Carlo_output"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    # Create a temporary directory to simulation runs
    simu_dir = project_path.parent / "simu_dir"
    if simu_dir.exists():
        shutil.rmtree(simu_dir, ignore_errors=True)
    zero_error_path = simu_dir / "run_ref" / "run.psx"
    zero_error_path.parent.mkdir(parents=True)
    doc.save(str(zero_error_path), [zero_error_chunk])

    iteration_part = partial(
        run_iteration,
        simu_dir=simu_dir,
        out_dir=out_dir,
        zero_error_chunk=zero_error_chunk,
        optimise_intrinsics=optimise_intrinsics,
        num_act_cam_orients=num_act_cam_orients,
    )

    if run_parallel:
        pool = Pool(parallell_processes)
        pool.map(iteration_part, range(num_randomisations))
    else:
        for run_idx in range(0, num_randomisations):
            iteration_part(run_idx)

    # Remove the temporary simulation directory
    shutil.rmtree(simu_dir, ignore_errors=True)

    # Read and process the output files
    mc_results.main(
        proj_dir=dir_path,
        pcd_ext="ply",
    )

    print("Finished")


if __name__ == "__main__":
    # Directory where output will be stored and active control file is saved.
    # The files will be generated in a sub-folder named "Monte_Carlo_output"
    dir_path = "data/rossia"
    project_name = "rossia_C_bis.psx"
    # dir_path = "data/calib_par"
    # project_name = "calib.psx"

    # Define how many times bundle adjustment (Metashape 'optimisation') will be carried out.
    num_randomisations = 1000
    run_parallel = False
    parallell_processes = 10

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

    montecarlo_simulation(
        dir_path=dir_path,
        project_name=project_name,
        num_randomisations=num_randomisations,
        run_parallel=run_parallel,
        parallell_processes=parallell_processes,
        optimise_intrinsics=optimise_intrinsics,
        pts_offset=pts_offset,
    )
