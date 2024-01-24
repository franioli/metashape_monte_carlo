import argparse
from pathlib import Path

import Metashape

from .workflow import cameras_from_bundler, create_new_project, import_markers

prm_to_optimize = {
    "f": True,
    "cx": True,
    "cy": True,
    "k1": True,
    "k2": True,
    "k3": True,
    "k4": False,
    "p1": True,
    "p2": True,
    "b1": False,
    "b2": False,
    "tiepoint_covariance": True,
}


def project_from_bundler(
    project_path: Path,
    images_dir: Path,
    bundler_file_path: Path,
    bundler_im_list: Path,
    marker_image_path: Path = None,
    marker_world_path: Path = None,
    marker_file_columns: str = "noxyz",
    prm_to_optimize: dict = {},
):
    image_list = list(images_dir.glob("*"))
    images = [str(x) for x in image_list if x.is_file()]

    doc = create_new_project(str(project_path), read_only=False)
    chunk = doc.chunk

    # Add photos to chunk
    chunk.addPhotos(images)
    cameras_from_bundler(
        chunk=chunk,
        fname=bundler_file_path,
        image_list=bundler_im_list,
    )

    # save project
    doc.read_only = False
    doc.save()

    # Import markers image coordinates
    if marker_image_path is not None:
        import_markers(
            marker_image_file=marker_image_path,
            chunk=chunk,
        )

    # Import markers world coordinates
    if marker_world_path is not None:
        chunk.importReference(
            path=str(marker_world_path),
            format=Metashape.ReferenceFormatCSV,
            delimiter=",",
            skip_rows=1,
            columns=marker_file_columns,
        )

    # optimize camera alignment
    if prm_to_optimize:
        chunk.optimizeCameras(
            fit_f=prm_to_optimize["f"],
            fit_cx=prm_to_optimize["cx"],
            fit_cy=prm_to_optimize["cy"],
            fit_k1=prm_to_optimize["k1"],
            fit_k2=prm_to_optimize["k2"],
            fit_k3=prm_to_optimize["k3"],
            fit_k4=prm_to_optimize["k4"],
            fit_p1=prm_to_optimize["p1"],
            fit_p2=prm_to_optimize["p2"],
            fit_b1=prm_to_optimize["b1"],
            fit_b2=prm_to_optimize["b2"],
            tiepoint_covariance=prm_to_optimize["tiepoint_covariance"],
        )

    # Export cameras and gcp residuals

    # save project
    doc.read_only = False
    doc.save()


def main(args):
    images_dir = Path(args.image_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist.")
    bundler_file_path = Path(args.bundler_file)
    if not bundler_file_path.exists():
        raise FileNotFoundError(f"Bundler file {bundler_file_path} does not exist.")
    bundler_im_list = Path(args.bundler_im_list)
    if not bundler_im_list.exists():
        raise FileNotFoundError(f"Bundler image list {bundler_im_list} does not exist.")

    project_path = Path(args.project_path)
    project_dir = project_path.parent

    if not project_dir.exists():
        project_dir.mkdir(parents=True)

    if project_path.suffix != ".psx":
        project_path = project_path.name + ".psx"

    project_from_bundler(
        project_path=project_path,
        images_dir=images_dir,
        bundler_file_path=bundler_file_path.resolve(),
        bundler_im_list=bundler_im_list.resolve(),
        prm_to_optimize=prm_to_optimize,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge all COLMAP databases of the input folder."
    )

    parser.add_argument(
        "--project_path",
        type=str,
        help="Path to the Metashape project file (.psx).",
        required=True,
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to the images directory.",
        required=True,
    )
    parser.add_argument(
        "bundler_file",
        type=str,
        help="Path to the bundler file.",
        required=True,
    )
    parser.add_argument(
        "bundler_im_list",
        type=str,
        help="Path to the bundler image list. If None is passed, it is assumed to be a file named 'bundler_list.txt' in the same directory of the bundler file.",
        default=None,
    )

    args = parser.parse_args()

    if args.bundler_im_list is None:
        args.bundler_im_list = args.bundler_file.parent / "bundler_list.txt"

    main(args)
