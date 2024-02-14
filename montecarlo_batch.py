import logging
from pathlib import Path

import Metashape

import mc_results

from .montecarlo import montecarlo_simulation

NaN = float("NaN")
logging.basicConfig(level=logging.INFO)

project_dir = "data/rossia"
project_dir = Path(project_dir)
project_paths = sorted(project_dir.glob("*.psx"))

num_randomisations = 2000
run_parallel = False
parallell_processes = 10
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
pts_offset = Metashape.Vector([0.0, 0.0, 0.0])

for project_path in project_paths:
    simu_dir = project_path / f"simulation_{project_path.stem}"
    montecarlo_simulation(
        project_path=project_path,
        simu_dir=simu_dir,
        num_randomisations=num_randomisations,
        run_parallel=run_parallel,
        parallell_processes=parallell_processes,
        optimise_intrinsics=optimise_intrinsics,
        pts_offset=pts_offset,
    )
    mc_results.main(
        proj_dir=simu_dir,
        pcd_ext="ply",
    )
