import csv
import math
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Union

import Metashape
import metashapelib as mslib
import numpy as np
from metashapelib import montecarlo as mc

logger = mslib.setup_logger(name="MC", log_level="DEBUG")

if not mslib.check_license():
    raise Exception(
        "No licence found. Please check that you linked your license (floating or standalone) wih the Metashape python module."
    )
backward_compatibility = mslib.backward_compatibility()

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


def montecarlo_simulation(
    project_path: Path,
    num_randomisations: int,
    simu_dir: Path = None,
    run_parallel: bool = False,
    workers: int = 10,
    resume_sumulations_from: int = -1,
    optimise_intrinsics: Dict = default_intrinsics_optim,
    pts_offset: Union[List, np.ndarray, Metashape.Vector] = [NaN, NaN, NaN],
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
    project_path = Path(project_path)
    if not project_path.exists():
        raise FileNotFoundError(f"File {project_path} does not exist.")
    if not project_path.suffix == ".psx":
        raise ValueError(f"File {project_path} is not a Metashape project file.")
    if not isinstance(pts_offset, Metashape.Vector):
        pts_offset = Metashape.Vector(pts_offset)

    # If the resume_sumulations_from counter is negative, initialise the simulation. If it is positive, skip it and resume the simulation from the specified run number.
    if resume_sumulations_from < 0:
        mc.initialise_simulation(
            project_path=project_path,
            simu_dir=simu_dir,
            pts_offset=pts_offset,
            optimise_intrinsics=optimise_intrinsics,
            skip_optimisation=True,
        )

    # Check the initialisation of the simulation
    if not mc.check_initialisation(simu_dir):
        raise RuntimeError("Cannot resume simulation. Initialisation failed.")
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
        [simu_dir] * act_num_randomisations,
        [optimise_intrinsics] * act_num_randomisations,
        [list(pts_offset)] * act_num_randomisations,
    )

    # Run the simulationsparse_pts_reference_cov
    logger.info("Starting Monte Carlo simulation...")
    if run_parallel:
        with Pool(workers) as pool:
            results = pool.imap(
                mc.run_iteration,
                iterable_parms,
            )
            logger.info("Succeded iterations:")
            for value in results:
                print("*", end=" ")
                if not value:
                    print("_", end=" ")
            print("")

    else:
        results = list(map(mc.run_iteration, iterable_parms))

    logger.info("Simulation completed.")

    return results


if __name__ == "__main__":
    # Directory where output will be stored and active control file is saved.
    # The files will be generated in a sub-folder named "Monte_Carlo_output"
    ref_project_path = "data/rossia/rossia_gcp_aat.psx"
    simu_name = "simulation_test"
    ref_project_path = "data/belvedere2023/belvedere2023.psx"
    simu_name = "simulation_belvedere2023"

    # Define how many times bundle adjustment (Metashape 'optimisation') will be carried out.
    num_randomisations = 5
    run_parallel = False
    workers = 5
    # NOTE: Keep the number of workers low if are running on a single CPU as the Metashape bundle adjustment is already multi-threaded and will use all available cores. If this is set too high, it will slow down the simulation and some runs may stuck.

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
        "p1": False,
        "p2": False,
        "tiepoint_covariance": True,
    }

    # The offset will be subtracted from point coordinates.
    # e.g.  pts_offset = [266000, 4702000, 0] for UTM coordinates
    pts_offset = [NaN, NaN, NaN]
    # pts_offset = [0.0, 0.0, 0.0]

    ref_project_path = Path(ref_project_path)
    simu_dir = ref_project_path.parent / simu_name

    # Run the Monte Carlo simulation
    montecarlo_simulation(
        project_path=ref_project_path,
        num_randomisations=num_randomisations,
        simu_dir=simu_dir,
        run_parallel=run_parallel,
        workers=workers,
        resume_sumulations_from=resume_sumulations_from,
        optimise_intrinsics=optimise_intrinsics,
        pts_offset=pts_offset,
    )
