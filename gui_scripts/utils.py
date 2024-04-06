import Metashape


def find_camera_by_label(chunk: Metashape.Chunk, label: str):
    cameras = chunk.cameras
    for cam in cameras:
        if cam.label == label:
            return cam
    print(f"camera {label} not found")
