import os
import time
import warnings
import argparse
from torch.utils.data import Dataset, DataLoader
from FlagEmbedding import BGEM3FlagModel

warnings.simplefilter(action="ignore", category=FutureWarning)


class SummaryDataset(Dataset):
    def __init__(self, folder_path, existing_files):
        self.folder_path = folder_path
        self.file_names = [
            f
            for f in os.listdir(folder_path)
            if f.endswith(".txt") and f not in existing_files
        ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]
        filepath = os.path.join(self.folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            summary = f.read().strip()
        return summary, filename  # Return filename instead of full path


import numpy as np
import h5py
from tqdm import tqdm


def process_folder(args):
    output_file = f"embeddings_{os.path.basename(args.input_folder)}.h5"

    # Check for existing embeddings
    existing_files = set()
    mode = "a"  # Default to append mode
    current_size = 0  # Initialize current_size to 0

    if os.path.exists(output_file):
        with h5py.File(output_file, "r") as f:
            if "filepaths" in f:
                existing_files = set(
                    os.path.basename(filepath.decode("utf-8"))
                    for filepath in f["filepaths"][:]
                )
            if "dense_embeddings" in f and "colbert_embeddings" in f:
                if f["dense_embeddings"].chunks is None:
                    print(
                        "Existing datasets are not chunked. Creating a new file with chunked datasets."
                    )
                    mode = "w"  # Switch to write mode to create a new file
                else:
                    current_size = f["dense_embeddings"].shape[
                        0
                    ]  # Set current_size if file exists

    dataset = SummaryDataset(args.input_folder, existing_files)
    if len(dataset) == 0:
        print("No new files to process.")
        return

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )

    new_dense_embeddings = []
    new_colbert_embeddings = []
    new_filenames = []

    for summaries, filenames in tqdm(dataloader,
                                     desc="Processing new summaries"):
        outputs = model.encode(
            summaries,
            batch_size=len(summaries),
            max_length=args.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        new_dense_embeddings.append(outputs["dense_vecs"])
        new_colbert_embeddings.extend(
            outputs["colbert_vecs"]
        )  # Extend instead of append
        new_filenames.extend(filenames)

    if not new_dense_embeddings or not new_colbert_embeddings:
        print("No new embeddings were generated. Exiting.")
        return

    new_dense_embeddings = np.concatenate(new_dense_embeddings, axis=0)

    with h5py.File(output_file, mode) as f:
        if "dense_embeddings" not in f or "colbert_embeddings" not in f or mode == "w":
            # Create new datasets
            chunk_size = min(
                1000, new_dense_embeddings.shape[0]
            )  # Adjust chunk size as needed
            f.create_dataset(
                "dense_embeddings",
                data=new_dense_embeddings,
                maxshape=(None, new_dense_embeddings.shape[1]),
                chunks=(chunk_size, new_dense_embeddings.shape[1]),
            )

            # Create a variable-length dataset for ColBERT embeddings
            colbert_dtype = h5py.special_dtype(vlen=np.dtype("float32"))
            f.create_dataset(
                "colbert_embeddings",
                (len(new_colbert_embeddings),),
                dtype=colbert_dtype,
                maxshape=(None,),
                chunks=(chunk_size,),
            )

            f.create_dataset(
                "filepaths",
                data=new_filenames,
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=h5py.special_dtype(vlen=str),
            )

            current_size = 0  # Reset current_size to 0 for new datasets
        else:
            # Append to existing datasets
            current_size = f["dense_embeddings"].shape[0]
            new_size = current_size + new_dense_embeddings.shape[0]

            f["dense_embeddings"].resize(
                (new_size, new_dense_embeddings.shape[1]))
            f["dense_embeddings"][current_size:] = new_dense_embeddings

            f["colbert_embeddings"].resize(
                (f["colbert_embeddings"].shape[0] + len(
                    new_colbert_embeddings),)
            )
            f["filepaths"].resize(
                (f["filepaths"].shape[0] + len(new_filenames),))

        # Store ColBERT embeddings
        for i, emb in enumerate(new_colbert_embeddings):
            f["colbert_embeddings"][
                current_size + i
                ] = emb.flatten()  # Store as 1D array

        f["filepaths"][current_size:] = new_filenames

    print(f"Embeddings saved to {output_file}")
    print(f"Processed {len(new_filenames)} new files.")
    print(f"Dense embedding shape: {new_dense_embeddings.shape}")
    print(f"ColBERT embeddings stored as variable-length arrays.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BGE-M3 Single Folder Embedding System for One GPU (Incremental)"
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        help="Path to the folder containing summary txt files.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length for the model.",
    )
    args = parser.parse_args()

    start_time = time.time()
    process_folder(args)
    total_time = time.time() - start_time
    print(
        f"Total time to process and embed summaries: {total_time:.4f} seconds.")
