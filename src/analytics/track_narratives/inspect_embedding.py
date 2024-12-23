"""
Get Global statistics and Visualization of embeddings
"""
import h5py
import numpy as np
import argparse

# Import visualization libraries only if needed
def import_viz_libraries():
    global plt, TSNE, PCA
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA


def load_embeddings(file_path):
    with h5py.File(file_path, "r") as f:
        embeddings = f["embeddings"][:]
        filepaths = f["filepaths"][:]
    return embeddings, filepaths


def summarize_embeddings(embeddings, filepaths, visualize=False):
    total_embeddings = len(embeddings)
    print(f"Total number of embeddings: {total_embeddings}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Basic statistics
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms summary:")
    print(f"  Min: {embedding_norms.min():.4f}")
    print(f"  Max: {embedding_norms.max():.4f}")
    print(f"  Mean: {embedding_norms.mean():.4f}")
    print(f"  Median: {np.median(embedding_norms):.4f}")
    print(f"  Std Dev: {np.std(embedding_norms):.4f}")
    print(f"  Shape: {embedding_norms.shape}")

    if visualize:
        import_viz_libraries()
        # Dimensionality reduction for quick overview
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title(
            f"PCA visualization of embeddings (2D)\nTotal embeddings: {total_embeddings}"
        )
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.show()


def view_sample_embeddings(embeddings, filepaths, num_samples=5):
    print(f"\nSample of {num_samples} embeddings and their corresponding files:")
    indices = np.random.choice(len(embeddings), num_samples, replace=False)
    for idx in indices:
        print(f"\nFile: {filepaths[idx].decode('utf-8')}")
        print(f"Embedding (first 10 dimensions): {embeddings[idx][:10]}...")


def visualize_tsne(
    embeddings, filepaths, num_samples=1000, perplexity=30, n_iter=1000, file_path=None
):
    import_viz_libraries()
    if len(embeddings) > num_samples:
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings_sample = embeddings[indices]
        filepaths_sample = filepaths[indices]
    else:
        embeddings_sample = embeddings
        filepaths_sample = filepaths

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_sample)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=range(len(embeddings_2d)),
        cmap="viridis",
        alpha=0.6,
    )
    plt.colorbar(scatter)
    plt.title(
        f"t-SNE visualization of embeddings (perplexity={perplexity}, n_iter={n_iter})"
    )
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")

    annot = plt.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = filepaths_sample[ind["ind"][0]].decode("utf-8")
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == plt.gca():
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                plt.draw()
            else:
                if vis:
                    annot.set_visible(False)
                    plt.draw()

    plt.gcf().canvas.mpl_connect("motion_notify_event", hover)
    plt.savefig(f"tsne_{file_path}.pdf")


def main(file_path, num_samples, perplexity, n_iter, visualize):
    embeddings, filepaths = load_embeddings(file_path)
    summarize_embeddings(embeddings, filepaths, visualize)
    view_sample_embeddings(embeddings, filepaths, num_samples)
    if visualize:
        visualize_tsne(
            embeddings, filepaths, num_samples, perplexity, n_iter, file_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and visualize embeddings from H5 file"
    )
    parser.add_argument("file_path", help="Path to the H5 file containing embeddings")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of random samples to display",
    )
    parser.add_argument(
        "--perplexity", type=float, default=30, help="Perplexity parameter for t-SNE"
    )
    parser.add_argument(
        "--n_iter", type=int, default=1000, help="Number of iterations for t-SNE"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization of embeddings"
    )
    args = parser.parse_args()

    main(args.file_path, args.num_samples, args.perplexity, args.n_iter, args.visualize)
