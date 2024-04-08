from pathlib import Path

import metashapelib as mslib
import numpy as np

logger = mslib.getlogger(name="metashapelib", log_level="DEBUG")
NaN = np.nan

# Directory where output will be stored and active control file is saved.
# The files will be generated in a sub-folder named "Monte_Carlo_output"
project_dir = Path("/mnt/phd/metashapelib/data/belv_stereo/").resolve()
ref_project_path = project_dir / "2022-07-28_15-02-49.psx"
simu_name = f"simulation_{ref_project_path.stem}_allioest"

# Define how many times bundle adjustment (Metashape 'optimisation') will be carried out.
num_randomisations = 400

# Run the Monte Carlo simulation in parallel
# NOTE: Keep the number of workers low if are running on a single CPU as the Metashape bundle adjustment is already multi-threaded and will use all available cores. If this is set too high, it will slow down the simulation and some runs may stuck.
run_parallel = True
workers = 12

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
    "k3": False,
    "k4": False,
    "p1": True,
    "p2": True,
    "tiepoint_covariance": True,
}

# The offset will be subtracted from point coordinates.
# e.g.  pts_offset = [266000, 4702000, 0] for UTM coordinates
# pts_offset = [NaN, NaN, NaN]
pts_offset = [0.0, 0.0, 0.0]

ref_project_path = Path(ref_project_path)
simu_dir = ref_project_path.parent / simu_name

# Run the Monte Carlo simulation
mslib.montecarlo.run_simulation(
    project_path=ref_project_path,
    num_randomisations=num_randomisations,
    simu_dir=simu_dir,
    run_parallel=run_parallel,
    workers=workers,
    resume_sumulations_from=resume_sumulations_from,
    optimise_intrinsics=optimise_intrinsics,
    pts_offset=pts_offset,
)
