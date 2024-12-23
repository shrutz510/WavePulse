"""
Update existing embeddings or merge multiple vector databases.
"""
import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm


def merge_embeddings(input_files, output_file):
    all_dense_embeddings = []
    all_filepaths = []

    # First pass: determine total sizes and shapes
    total_dense = 0
    dense_dim = None
    skipped_files = []

    for input_file in tqdm(input_files, desc="Calculating sizes"):
        try:
            with h5py.File(input_file, "r") as f:
                if "dense_embeddings" in f:
                    total_dense += f["dense_embeddings"].shape[0]
                    if dense_dim is None:
                        dense_dim = f["dense_embeddings"].shape[1]
                else:
                    print(
                        f"Warning: File {input_file} does not contain expected datasets. Skipping."
                    )
                    skipped_files.append(input_file)
        except Exception as e:
            print(f"Error processing file {input_file}: {str(e)}")
            skipped_files.append(input_file)

    if not total_dense or dense_dim is None:
        print("No valid data found in input files.")
        return

    # Create the output file with pre-allocated datasets
    with h5py.File(output_file, "w") as out_f:
        dense_dset = out_f.create_dataset(
            "dense_embeddings", shape=(total_dense, dense_dim), dtype=np.float32
        )
        filepath_dset = out_f.create_dataset(
            "filepaths", shape=(total_dense,), dtype=h5py.special_dtype(vlen=str)
        )

        # Second pass: copy data
        dense_offset = 0
        for input_file in tqdm(input_files, desc="Merging files"):
            if input_file in skipped_files:
                continue
            try:
                with h5py.File(input_file, "r") as f:
                    dense_count = f["dense_embeddings"].shape[0]

                    dense_dset[dense_offset : dense_offset + dense_count] = f[
                        "dense_embeddings"
                    ][:]
                    filepath_dset[dense_offset : dense_offset + dense_count] = f[
                        "filepaths"
                    ][:]

                    dense_offset += dense_count
            except Exception as e:
                print(f"Error processing file {input_file}: {str(e)}")

    print(f"Merged embeddings saved to {output_file}")
    print(f"Total dense embeddings: {total_dense}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files due to errors or incompatibility.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge Dense Embeddings from Multiple Files"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Directory containing the embedding H5 files to merge.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="merged_dense_embeddings.h5",
        help="Output file for merged embeddings.",
    )
    args = parser.parse_args()

    input_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".h5")
    ]

    if not input_files:
        print("No H5 files found in the input directory.")
    else:
        merge_embeddings(input_files, args.output_file)
