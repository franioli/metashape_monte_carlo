import shutil
from pathlib import Path

import dask
import numpy as np

from .mc_results import load_pcd, write_pcd


def generate_random_data(directory, npcd, npoints, noise_std):
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    def random_points3d_uniform(
        npoints=1000,
        xlim=(100, 200),
        ylim=(100, 200),
        zlim=(0, 1),
        datatype=np.float32,
    ):
        rng = np.random.default_rng(seed=0)
        x = rng.uniform(xlim[0], xlim[1], npoints)
        y = rng.uniform(ylim[0], ylim[1], npoints)
        z = rng.uniform(zlim[0], zlim[1], npoints)
        return np.stack([x, y, z], axis=1).astype(datatype)

    def generate_simulated_pcd(pcd_path, ref_pcd_path, noise_std):
        rng = np.random.default_rng()
        xyz = load_pcd(ref_pcd_path)
        xyz += rng.normal(0, noise_std, xyz.shape)
        return write_pcd(xyz, pcd_path)

    # generate random reference pcd
    ref_pcd_path = directory / "sparse_pts_reference.ply"
    xyz = random_points3d_uniform(npoints)
    write_pcd(xyz, ref_pcd_path)

    # generate random pcds starting from reference and adding noise
    output_directory = directory / "Monte_Carlo_output"
    output_directory.mkdir()

    if npcd > 1000:
        delayed_tasks = []
        for i in range(npcd):
            path = output_directory / f"{i:04}_pts.ply"
            generate_simulated_pcd(path, ref_pcd_path, noise_std)

            delayed_tasks.append(
                dask.delayed(generate_simulated_pcd)(path, ref_pcd_path, noise_std)
            )

        dask.compute(*delayed_tasks)
    else:
        for i in range(npcd):
            path = output_directory / f"{i:04}_pts.ply"
            generate_simulated_pcd(path, ref_pcd_path, noise_std)


if __name__ == "__main__":
    num_pcd = 1000
    num_points = 100000
    noise_std = 0.01

    # Generate Random point cloud as test data
    proj_dir = Path("data/test")
    generate_random_data(proj_dir, num_pcd, num_points, noise_std)
