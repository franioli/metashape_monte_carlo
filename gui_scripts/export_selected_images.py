import shutil

import Metashape

# Checking compatibility
compatible_major_version = ["1.8", "2.0", "2.1"]
found_major_version = ".".join(Metashape.app.version.split(".")[:2])
if found_major_version not in compatible_major_version:
    raise Exception(
        "Incompatible Metashape version: {} != {}".format(
            found_major_version, compatible_major_version
        )
    )

doc = Metashape.app.document


def main():
    dest_path = Metashape.app.getExistingDirectory("Select output folder:")
    if not dest_path:
        return
    print("Exporting selected cameras ...")
    chunk = doc.chunk
    for camera in chunk.cameras:
        if camera.selected:
            print(camera.photo.path)
            shutil.copy2(camera.photo.path, dest_path)

    print(f"Images exported to {dest_path}.\n")


Metashape.app.addMenuItem("Scripts/Export selected images", main)
