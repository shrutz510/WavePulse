"""
Run this if you have erroneous vector database
The code loads the embeddings (whichever exists) and then saves them after reshaping.
"""
import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm


def fix_h5_file(input_file, output_file):
    try:
        with h5py.File(input_file, "r") as f:
            # Check if required datasets exist
            if (
                "dense_embeddings" not in f
                or "colbert_embeddings" not in f
                or "filepaths" not in f
            ):
                raise KeyError("Missing required datasets")

            dense_embeddings = f["dense_embeddings"][:]
            colbert_embeddings = f["colbert_embeddings"][:]
            filepaths = f["filepaths"][:]

            # Check and fix ColBERT embeddings shape
            if len(colbert_embeddings.shape) == 1:
                # Reshape to 2D if it's 1D
                colbert_dim = colbert_embeddings[0].shape[0]
                colbert_embeddings = np.array(
                    [emb.reshape(-1, colbert_dim) for emb in colbert_embeddings]
                )
            elif len(colbert_embeddings.shape) == 3:
                # Flatten if it's 3D
                colbert_embeddings = colbert_embeddings.reshape(
                    colbert_embeddings.shape[0], -1
                )

        # Write fixed data to new file
        with h5py.File(output_file, "w") as f_out:
            f_out.create_dataset("dense_embeddings", data=dense_embeddings)
            f_out.create_dataset("colbert_embeddings", data=colbert_embeddings)
            f_out.create_dataset("filepaths", data=filepaths)

        print(f"Fixed file saved as {output_file}")
        return True
    except Exception as e:
        print(f"Error fixing file {input_file}: {str(e)}")
        return False


def fix_all_h5_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".h5")]

    fixed_files = []
    for filename in tqdm(input_files, desc="Fixing files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"fixed_{filename}")
        if fix_h5_file(input_path, output_path):
            fixed_files.append(output_path)

    return fixed_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix H5 Files Before Merging")
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Directory containing the original H5 files.",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Directory to save fixed H5 files."
    )
    args = parser.parse_args()

    fixed_files = fix_all_h5_files(args.input_dir, args.output_dir)
    print(f"Fixed {len(fixed_files)} files. You can now merge these fixed files.")
