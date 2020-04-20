import os
import time
import argparse
import pickle
import concurrent.futures as concurr

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()  # No need for session to be created. Function instances are run immediately.

from waymo_open_dataset import dataset_pb2 as open_dataset
from google.cloud import storage


# CONFIG
project = "Waymo3DObjectDetection"
bucket_name = "waymo_open_dataset_v_1_2_0_individual_files"
suffix = ".tfrecord"
data_destination = os.getcwd() + "/data/"
download_batch_size = 1


def download_blob(blob, c):
    """
    blob = single file name
    c = file counter
    """
    fname = f"{data_destination}blob_{c}{suffix}"
    blob.download_to_filename(fname)
    return fname


def strip_frame(frame, idx, blob_idx):
    """Strip frame from garbage such as LIDAR data"""

    cam_dict = {}
    for i, camera in enumerate(
        ["FRONT", "FRONT_LEFT", "SIDE_LEFT", "FRONT_RIGHT", "SIDE_RIGHT"]
    ):
        cam_dict[camera] = {}
        cam_dict[camera]["image"] = frame.images[i].image
        cam_dict[camera]["velocity"] = frame.images[i].velocity
        cam_dict[camera]["labels"] = frame.camera_labels[i]

        cam_dict[camera]["context"] = {
            "stats": frame.context.stats,
            "name": frame.context.name,
            "blob_idx": blob_idx,
            "time_frame_idx": idx,
        }
    return cam_dict


def save_frames(frames, blob_idx, dataset="training"):
    """Save frames into pickle format. To preprocess later"""
    for frame_idx, frame in enumerate(frames):
        for camera, camera_dict in frame.items():
            with open(
                f"{data_destination}{dataset}/{camera}/blob_{blob_idx}_frame_{frame_idx}.pickle",
                "wb",
            ) as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(camera_dict, f, pickle.HIGHEST_PROTOCOL)
    return None


def load_frame(frame_idx, blob_idx, dataset="training"):
    with open(f"{data_destination}{dataset}/blob_{blob_idx}.pickle", "rb") as f:
        # Load the 'data' dictionary using the highest protocol available.
        return pickle.load(f, pickle.HIGHEST_PROTOCOL)


# Retrieve frames from selected files to download
def get_and_strip_frames_from_one_blob(downloaded_blob, blob_idx):
    # Load into tf record dataset
    dataset = tf.data.TFRecordDataset(downloaded_blob, compression_type="")
    frames = []
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # Function to strip away LIDAR and other garbage from frame
        frame = strip_frame(frame, idx, blob_idx)
        frames.append(frame)
    return frames


def download_process_save_1_blob(blob, blob_idx, dataset="training"):
    """Like dem descriptive func names eh?"""

    print(f"Downloading blob_{blob_idx}")
    blob_fname = download_blob(blob, blob_idx)

    print(f"Getting and stripping all frames from blob_{blob_idx}")
    frames = get_and_strip_frames_from_one_blob(blob_fname, blob_idx)

    print(f"Saving frames for blob {blob_idx}")
    save_frames(frames, blob_idx, dataset)

    print(f"No longer need tfrecord blob_{blob_idx}. Deleting now.")
    os.remove(f"data/blob_{blob_idx}.tfrecord")

    return f"blob_{blob_idx}"


def run(dataset="training", num_workers=2):
    start = time.time()
    downloaded_blobs = []

    thread_iterable = ((blob, blob_idx, dataset) for blob_idx, blob in enumerate(blobs))

    with concurr.ThreadPoolExecutor(max_workers=num_workers) as executor:

        results = executor.map(
            lambda args: download_process_save_1_blob(*args), thread_iterable
        )
        for r in results:
            print(f"\n Time elapsed {time.time() - start}")
            downloaded_blobs.append(r)

    end = time.time()
    print(f"Total time taken {end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input args")
    parser.add_argument("dataset", type=str, nargs="+", help="training/validation/test")
    parser.add_argument(
        "num_workers", type=str, nargs="+", help="How many concurrent threads to run"
    )
    args = parser.parse_args().__dict__
    run(**args)
