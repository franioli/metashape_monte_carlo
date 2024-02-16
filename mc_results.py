import logging
import shutil
from pathlib import Path
from typing import List

import dask
import dask.array as da
import laspy
import matplotlib
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns
from matplotlib import pyplot as plt

from thirdparty import transformations as tf

matplotlib.use("agg")
logging.basicConfig(level=logging.INFO)


def load_pcd(pcd_path: Path) -> np.ndarray:
    return np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points).astype(np.float32)


def write_pcd(
    xyz: np.ndarray,
    path: Path,
) -> bool:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(str(path), pcd)
    return True


def lazy_load_pcd_stack(pcd_list):
    # load_pcd the first pcd (assume rest are same shape/dtype)
    sample = load_pcd(pcd_list[0])

    # Build a list of lazy dask arrays
    arrays = [
        da.from_delayed(
            dask.delayed(load_pcd)(path),
            dtype=sample.dtype,
            shape=sample.shape,
        )
        for path in pcd_list
    ]
    # Stack all point coordinates into a 3D dask array
    stack = da.stack(arrays, axis=0)
    stack = stack.rechunk()
    return stack


def load_pcd_stack(pcd_list):
    arrays = [load_pcd(path) for path in pcd_list]
    stack = np.stack(arrays, axis=0)
    return stack


def write_pcd_las(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rgb: np.ndarray = None,
    offset=np.array([0.0, 0.0, 0.0]),
    precision: float = 1e-5,
    **kwargs,
) -> bool:
    """
    Write point cloud data to a LAS file.

    Args:
        path (Path): The path to save the LAS file.
        x (np.ndarray): Array of x-coordinates of the points.
        y (np.ndarray): Array of y-coordinates of the points.
        z (np.ndarray): Array of z-coordinates of the points.
        rgb (np.ndarray, optional): Array of RGB values for each point. Must be a 3xn numpy array of 16-bit unsigned integers. Defaults to None.
        offset (np.ndarray, optional): Offset to be added to the coordinates.Defaults to np.array([0.0, 0.0, 0.0]).
        precision (float, optional): Precision of the coordinates. Defaults to 1e-5.
        **kwargs: Additional keyword arguments to be written as extra scalar fields in the LAS file.

    Returns:
        bool: True if writing to the LAS file is successful, False otherwise.

    Raises:
        TypeError: If the rgb argument is not a 3xn numpy array of 16-bit unsigned integers.

    Note:
        - If the path does not end with '.las', '.las' will be appended to the file name.
        - If the rgb argument is provided, it must be a 3xn numpy array of 16-bit unsigned integers.
        - The header of the LAS file will be set to version "1.4" and point format "2".
        - Additional keyword arguments (**kwargs) will be written as extra scalar fields in the LAS file.
    """
    path = Path(path)
    if path.suffix != ".las":
        path = path.parent / (path.name + ".las")

    path.parent.mkdir(parents=True, exist_ok=True)

    if rgb is not None:
        if not isinstance(rgb, np.ndarray) or rgb.shape[1] != 3:
            raise TypeError(
                "Inval rgb argument. It must a 3xn numpy array of 16bit unsigned int"
            )
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.uint16)
        elif rgb.dtype != np.uint16:
            raise TypeError(
                "Inval rgb argument. It must a 3xn numpy array of 16bit unsigned int"
            )

    header = laspy.LasHeader(version="1.4", point_format=3)
    header.add_extra_dims(
        [laspy.ExtraBytesParams(k, type=np.float32) for k in kwargs.keys()]
    )
    header.offsets = offset
    header.scales = [
        precision for _ in range(3)
    ]  # This is the precision of the coordinates

    with laspy.open(path, mode="w", header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(x.shape[0], header=header)
        point_record.x = x.astype(np.float32)
        point_record.y = y.astype(np.float32)
        point_record.z = z.astype(np.float32)
        if rgb is not None:
            point_record.red = rgb[:, 0]
            point_record.green = rgb[:, 1]
            point_record.blue = rgb[:, 2]

        for k, v in kwargs.items():
            setattr(point_record, k, v)

        writer.write_points(point_record)

    return True


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


def make_2D_scatter_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    color_values: np.ndarray,
    marker: str = "o",
    markersize: int = 1,
    ax: plt.Axes = None,
    title: str = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    colorbar: bool = False,
    colorbar_label: str = "Color Bar Label",
    cmap: str = "viridis",
    colorbar_limits: List = None,
) -> plt.Axes:
    """
    Create a 2D scatter plot.

    Args:
        x_values (np.ndarray): The x-coordinates of the points.
        y_values (np.ndarray): The y-coordinates of the points.
        color_values (np.ndarray): The values to use for coloring the points.
        marker (str, optional): Marker style. Defaults to "o".
        markersize (int, optional): Marker size. Defaults to 1.
        ax (plt.Axes, optional): Axes object to plot on. If None, a new figure will be created. Defaults to None.
        title (str, optional): Title for the plot. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to "X".
        ylabel (str, optional): Label for the y-axis. Defaults to "Y".
        colorbar (bool, optional): Whether to add a colorbar. Defaults to False.
        cmap (str, optional): The colormap to use for coloring the points. Defaults to "viridis".
        Colorbar_label (str, optional): Label for the colorbar. Defaults to "Color Bar Label".
        colorbar_limits (tuple, optional): Tuple containing the lower and upper limits of the colorbar. Defaults to None.

    Returns:
        plt.Axes: The Axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    if colorbar_limits:
        vmin, vmax = colorbar_limits
    else:
        vmin, vmax = color_values.min(), color_values.max()

    ax.margins(0.05)
    plot = ax.scatter(
        x_values,
        y_values,
        c=color_values,
        marker=marker,
        s=markersize,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    if colorbar:
        cbar = plt.colorbar(plot, ax=ax)
        cbar.set_label(colorbar_label)

    return ax


def make_3D_scatter_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    color_values: np.ndarray,
    marker: str = "o",
    markersize: int = 1,
    out_path: Path = "3D_scatter_plot.html",
    title: str = "3D Scatter Plot",
    colorbar: bool = False,
    colorbar_label: str = "Color Bar Label",
    colorbar_limits: List = None,
):
    """
    Create a 3D scatter plot using Plotly.

    Args:
        x_values (np.ndarray): The x-coordinates of the points.
        y_values (np.ndarray): The y-coordinates of the points.
        z_values (np.ndarray): The z-coordinates of the points.
        color_values (np.ndarray): The values to use for coloring the points.
        marker (str, optional): Marker style. Defaults to "o".
        markersize (int, optional): Marker size. Defaults to 1.
        out_path (Path, optional): The path where the plot will be saved. Defaults to "3D_scatter_plot.html".
        title (str, optional): Title for the plot. Defaults to "3D Scatter Plot".
        colorbar (bool, optional): Whether to add a colorbar. Defaults to False.
        colorbar_label (str, optional): Label for the colorbar. Defaults to "Color Bar Label".
        colorbar_limits (tuple, optional): Tuple containing the lower and upper limits of the colorbar. Defaults to None.
    """
    # Create trace for scatter plot
    trace = go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="markers",
        marker=dict(
            size=markersize,
            color=color_values,
            colorscale="viridis",
            opacity=0.8,
            colorbar=dict(
                title=colorbar_label,
                tickvals=np.linspace(color_values.min(), color_values.max(), 5),
                ticktext=[
                    f"{val:.2f}"
                    for val in np.linspace(color_values.min(), color_values.max(), 5)
                ],
            ),
        ),
    )

    # Create layout
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)

    # Save plot to HTML file
    pio.write_html(fig, str(out_path))


def make_precision_plot(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    out_path: Path = "estimated_precision.png",
    point_size: int = 1,
    scale_fct: float = 1.0,
    clim: List[List] = None,
    make_3D_plot: bool = False,
):
    """
    Creates a 2D planar scatter plot of the points, colored by the standard deviation of the coordinates.

    Args:
        x (np.ndarray): The x-coordinates of the points.
        y (np.ndarray): The y-coordinates of the points.
        z (np.ndarray): The z-coordinates of the points.
        sx (np.ndarray): The standard deviation of the x-coordinates.
        sy (np.ndarray): The standard deviation of the y-coordinates.
        sz (np.ndarray): The standard deviation of the z-coordinates.
        out_path (Path, optional): The path where the plot image will be saved. Defaults to "estiamted_precision.png".
        point_size (int, optional): The size of the points in the scatter plot. Defaults to 1.
        scale_fct (int, optional): The factor by which the standard deviation values are scaled for coloring. Defaults to 1000.
    """

    if clim is not None:
        # TODO: manage better the colorbar limits!
        if len(clim) != 1:
            [clim, clim, clim]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loop through each axis
    for i, (title, prec, lim) in enumerate(zip(["X", "Y", "Z"], [sx, sy, sz], clim)):
        ax = make_2D_scatter_plot(
            x,
            y,
            prec * scale_fct,
            ax=axes[i],
            markersize=point_size,
            colorbar=True,
            cmap="viridis",
            title=f"Precision {title} [mm]",
            colorbar_label="Standard Deviation",
            colorbar_limits=lim,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)

    if make_3D_plot:
        make_3D_scatter_plot(
            x,
            y,
            z,
            (sx**2 + sy**2 + sz**2) ** 0.5 * scale_fct,
            markersize=point_size,
            out_path=out_path.parent / (out_path.stem + ".html"),
            title="3D Scatter Plot",
            colorbar=True,
            colorbar_label="Standard Deviation",
            colorbar_limits=clim,
        )


def rmse(predicted: np.ndarray, reference: np.ndarray, axis: int = None) -> float:
    """
    Compute the root mean square error (RMSE) between two arrays.

    Args:
        predicted (np.ndarray): The predicted values.
        reference (np.ndarray): The reference values.

    Returns:
        float: The RMSE between the predicted and target values.
    """
    return np.sqrt(np.mean((predicted - reference) ** 2, axis=axis))


def compute_statistics(
    estimated: np.ndarray,
    reference: np.ndarray = None,
    make_plot: bool = True,
    figure_path: Path = "statistics.png",
):
    if reference is not None:
        diff = estimated - reference
    else:
        diff = estimated
    mean_diff = np.mean(diff, axis=0)
    std_diff = np.std(diff, axis=0)
    print("Mean difference [mm]:", mean_diff * 1000)
    print("Standard deviation of difference [mm]:", std_diff * 1000)
    if reference is not None:
        rmse_val = rmse(reference, estimated, axis=0)
        print("RMSE [mm]:", rmse_val * 1000)

    if make_plot:
        # Plot histogram of differences
        norms = np.linalg.norm(diff, axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].hist(
            norms,
            bins=20,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
            density=True,
        )
        axes[0].set_xlabel("Difference")
        axes[0].set_ylabel("Density")
        axes[0].grid(True)

        # Create a boxplot
        sns.boxplot(data=diff, palette="Set3", ax=axes[1])
        axes[1].set_xlabel("Axis")
        axes[1].set_ylabel("Difference")
        axes[1].grid(True)

        # Add mean and standard deviation as text to the plot
        mean_labels = [
            f"Mean {coord}: {val * 1000:.6f} mm"
            for coord, val in zip(["X", "Y", "Z"], mean_diff)
        ]
        std_labels = [
            f"Std Dev {coord}: {val * 1000:.6f} mm"
            for coord, val in zip(["X", "Y", "Z"], std_diff)
        ]
        if reference is not None:
            rmse_label = [
                f"RMSE {coord}: {val * 1000:.6f} mm"
                for coord, val in zip(["X", "Y", "Z"], rmse_val)
            ]
        else:
            rmse_label = []

        fig.text(
            0.8,
            0.5,
            "\n".join(mean_labels + std_labels + rmse_label),
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.5),
            transform=fig.transFigure,
        )
        fig.tight_layout(
            rect=[0, 0.1, 0.7, 0.9]
        )  # Adjust the layout to leave space for the text
        fig.savefig(figure_path, dpi=300)


def make_doming_plot(data: pd.DataFrame, fig_path: Path, print_stats: bool = False):
    if print_stats:
        for name, group in data.groupby("Enable"):
            print(name, group.describe())

    vmin, vmax = data["Z_error"].min(), data["Z_error"].max()
    group_labels = {
        1: "GCPs",
        0: "CPs",
    }
    marker_style = {
        1: "o",  # Enabled points
        0: "D",  # Disabled points
    }
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    # Upper plot: Scatter plot of points with Z_error as color
    axes[0].margins(0.05)
    legend_labels = []
    for enabled, group in data.groupby("Enable"):
        scatter = axes[0].scatter(
            group["X"],
            group["Y"],
            c=group["Z_error"],
            cmap="viridis",
            alpha=0.9,
            marker=marker_style[enabled],
            vmin=vmin,
            vmax=vmax,
        )
        legend_labels.append(group_labels[enabled])
    cbar = plt.colorbar(scatter, label="Z Error", ax=axes[0])
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_title("2D Scatter Plot of Control Points (CPs)")
    axes[0].set_aspect("equal")
    axes[0].legend(legend_labels)

    # Compute baricenter of GCPs
    gcp_baricenter = data[["X", "Y", "Z"]].mean()
    data["GCP_distance"] = np.linalg.norm(
        data[["X", "Y", "Z"]].values - gcp_baricenter.values, axis=1
    )

    # Lower plot: distance of each point from the baricenter vs Z error
    axes[1].margins(0.05)
    for name, group in data.groupby("Enable"):
        axes[1].scatter(
            group["GCP_distance"],
            group["Z_error"],
            marker=marker_style[name],
            c="red" if name == 1 else "blue",
            alpha=0.9,
        )
    axes[1].set_xlabel("Distance from GCP Baricenter")
    axes[1].set_ylabel("Z Error")
    axes[1].set_title("Distance from GCP Baricenter vs Z Error")
    axes[1].legend(legend_labels)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def main(
    proj_dir,
    pcd_ext: str = "ply",
    offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
    use_dask: bool = False,
    compute_full_covariance: bool = True,
    cov_ddof: int = 1,
):
    # Get pcd list
    pcd_dir = proj_dir / "Monte_Carlo_output"
    pcd_list = sorted(list(pcd_dir.glob(f"*.{pcd_ext}")))
    logging.info(f"Found {len(pcd_list)} pointclouds in {pcd_dir}")

    # Read reference point cloud from MC simulation
    ref_pcd_path = pcd_dir.parent / "sparse_pts_reference.ply"
    if ref_pcd_path.exists():
        ref_pcd = load_pcd(ref_pcd_path)
        logging.info(f"Loaded reference pointcloud from {ref_pcd_path}")
    else:
        ref_pcd = load_pcd(pcd_list[0])
        logging.info(
            "Reference pointcloud not found, using first pointcloud as reference"
        )

    # Build a lazy dask array of all pointclouds
    stack = lazy_load_pcd_stack(pcd_list)

    # Load the data and compute the mean and std with dask
    if use_dask:
        operations = [
            stack.mean(axis=0),
            stack.std(axis=0, ddof=cov_ddof),
        ]
        logging.info("Computing mean and std with dask...")
        mean, std = dask.compute(*operations)
    else:
        # Load all the data into memory and compute the mean and std
        stack = np.array([load_pcd(path) for path in pcd_list])
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0, ddof=cov_ddof)

    # Compute full covariance matrix for each point (note that all the pcd are loaded in memory at once here!)
    if compute_full_covariance:
        skip_cov = False
        try:

            def compute_covariance(points):
                return np.cov(points, rowvar=False, ddof=cov_ddof)

            if isinstance(stack, da.Array):
                stack = np.array(stack)
            np_cov = [compute_covariance(stack[:, i, :]) for i in range(stack.shape[1])]
            np.sqrt(np_cov[0].diagonal()) - std[0]
        except MemoryError:
            logging.error("Not enough memory to compute full covariance matrix")
            skip_cov = True
    else:
        skip_cov = True

    # # Add offset to the mean point coordinates
    mean += offset

    # Estimate a Helmert transformation between the mean pcd and the ref.
    T = tf.affine_matrix_from_points(
        mean.T, ref_pcd.T, shear=False, scale=True, usesvd=True
    )
    scale, _, angles, translation, _ = tf.decompose_matrix(T)
    scale_percent = scale.mean() - 1
    angles_deg = np.rad2deg(angles)
    logging.info(f"Translation: {translation*1000} mm")
    logging.info(f"Angles: {angles_deg} deg")
    logging.info(f"Scale: {scale_percent:.6}%")

    pts_homog = np.hstack([mean, np.ones((mean.shape[0], 1))]).T
    points_roto = ((T @ pts_homog).T)[:, :3]

    if not skip_cov:
        # Rotate covariance matrix and compute new standard deviation
        R = T[:3, :3]
        cov_roto = np.array([R @ cov @ R.T for cov in np_cov])
        std_roto = np.sqrt(np.array([np.diag(cov) for cov in cov_roto]))

    # Compute statistics
    compute_statistics(
        estimated=mean,
        reference=ref_pcd,
        make_plot=True,
        figure_path=proj_dir / "difference_stats.png",
    )

    # Compute statistics for the rototranslated points
    compute_statistics(
        estimated=points_roto,
        reference=ref_pcd,
        make_plot=True,
        figure_path=proj_dir / "difference_helmert_stats.png",
    )

    # Make a 2D precision plot
    scale_fct = 1e3
    clim_quantile = 0.95
    clim = [
        (
            np.floor(np.quantile(std[:, i], 1 - clim_quantile) * scale_fct / 2) * 2,
            np.ceil(np.quantile(std[:, i], clim_quantile) * scale_fct / 2) * 2,
        )
        for i in range(3)
    ]
    make_precision_plot(
        mean[:, 0],
        mean[:, 1],
        mean[:, 2],
        std[:, 0],
        std[:, 1],
        std[:, 2],
        proj_dir / "estimated_precision.png",
        scale_fct=scale_fct,
        clim=clim,
        make_3D_plot=True,
    )

    # Create a las pcd with laspy
    rgb = np.uint16(np.asarray(o3d.io.read_point_cloud(str(ref_pcd_path)).colors) * 255)
    scalar_fields = {
        "sx": std[:, 0],
        "sy": std[:, 1],
        "sz": std[:, 2],
    }
    write_pcd_las(
        proj_dir / "point_precision.las",
        mean[:, 0],
        mean[:, 1],
        mean[:, 2],
        rgb=rgb,
        **scalar_fields,
    )

    scalar_fields = {
        "sx": std_roto[:, 0],
        "sy": std_roto[:, 1],
        "sz": std_roto[:, 2],
    }
    write_pcd_las(
        proj_dir / "point_precision_helmert.las",
        points_roto[:, 0],
        points_roto[:, 1],
        points_roto[:, 2],
        rgb=rgb,
        **scalar_fields,
    )

    # Do Ground Control Analysis
    gc_files = sorted((proj_dir / "Monte_Carlo_output").glob("*_GC.txt"))

    # Make doming plot from ground control data for the first file
    file = gc_files[0]
    data = pd.read_csv(file, skiprows=1, header=0)
    fig_path = proj_dir / "doming_effect.png"

    make_doming_plot(data, fig_path, print_stats=False)

    # Compute summary statistics for all ground control files
    def get_rmse_max(file):
        data = pd.read_csv(file, skiprows=1, header=0)
        rmse_z = np.sqrt((data["Z_error"] ** 2).mean())
        max_z_err = data["Z_error"].abs().max()
        return rmse_z, max_z_err

    delayed_tasks = []
    for file in gc_files:
        delayed_tasks.append(dask.delayed(get_rmse_max)(file))
    res = dask.compute(*delayed_tasks)

    rmse_z, max_z_err = zip(*res)
    rmse_z = np.array(rmse_z)
    max_z_err = np.array(max_z_err)
    logging.info(f"Mean Z RMSE: {rmse_z.mean():.4f} m")
    logging.info(f"Max Z error: {max_z_err.mean():.4f} m")

    # Read cameras data

    # Make 2D precision plot with point precision from MS
    ms_ref = proj_dir / "sparse_pts_reference_cov.csv"
    if ms_ref.exists():
        data = np.genfromtxt(ms_ref, delimiter=",", skip_header=1)

        # Read coordinate offset used in MC simulations from disk
        off = np.loadtxt(proj_dir / "_coordinate_local_origin.txt")

        xyz = data[:, 1:4] - off
        rgb = data[:, 4:7]
        precision = data[:, 7:10]
        covariances = data[:, 10:]
        clim = [(0, 30) for _ in range(3)]
        make_precision_plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            precision[:, 0],
            precision[:, 1],
            precision[:, 2],
            proj_dir / "metashape_reference_precision.png",
            scale_fct=scale_fct,
            clim=clim,
        )
    else:
        logging.info("No reference precision computed from metashape data found")

    logging.info("Done")


if __name__ == "__main__":
    proj_dir = Path("data/rossia/simulation_rossia_4gcp")
    pcd_ext = "ply"
    compute_full_covariance = True
    use_dask = False
    cov_ddof = 1

    # Random data parameters
    generate_random_pcd = False
    num_pcd = 1000
    num_points = 100000
    noise_std = 0.01

    # Generate Random point cloud as test data
    if generate_random_pcd:
        proj_dir = Path("data/test")
        generate_random_data(proj_dir, num_pcd, num_points, noise_std)
        logging.info("Generated random point cloud data")

    main(
        proj_dir,
        pcd_ext=pcd_ext,
        use_dask=use_dask,
        compute_full_covariance=compute_full_covariance,
        cov_ddof=cov_ddof,
    )
