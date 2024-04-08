import os

import Metashape

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split(".")[:2])
if found_major_version != compatible_major_version:
    raise Exception(
        "Incompatible Metashape version: {} != {}".format(
            found_major_version, compatible_major_version
        )
    )


def write_markers_one_cam_per_file() -> None:
    """Write Marker image coordinates to csv file (one file per camera),
    named with camera label.
    Each file is formatted as follows:
    marker1, x, y
    marker2, x, y
    ...
    markerM, x, y

    """
    output_dir = Metashape.app.getExistingDirectory()
    doc = Metashape.app.document
    chunk = doc.chunk

    for camera in chunk.cameras:
        # Write header to file
        fname = os.path.join(output_dir, camera.label + ".csv")
        file = open(fname, "w")
        file.write("label,x,y\n")

        for marker in chunk.markers:
            projections = marker.projections  # list of marker projections
            marker_name = marker.label

            for cur_cam in marker.projections.keys():
                if cur_cam == camera:
                    x, y, _ = projections[cur_cam].coord

                    # subtract 0.5 px to image coordinates (metashape image RS)
                    x -= 0.5
                    y -= 0.5

                    # writing output to file
                    file.write(f"{marker_name},{x:.4f},{y:.4f}\n")
        file.close()

    print("All targets exported successfully")


label = "Scripts/Export markers by camera"
Metashape.app.addMenuItem(label, write_markers_one_cam_per_file)
