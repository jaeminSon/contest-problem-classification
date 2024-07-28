import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE


def load_vectors(dir_home):
    vectors = [np.load(path_embedding)
               for path_embedding in Path(dir_home).iterdir()]
    return np.stack(vectors, axis=0)


def spectral_clustering(data, n_clusters=20):
    similarity_matrix = np.dot(data, data.T)

    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed', assign_labels='cluster_qr')
    labels = spectral.fit_predict(similarity_matrix)

    return labels


if __name__ == "__main__":
    data = load_vectors("./Category")
    labels = spectral_clustering(data)

    tsne = TSNE(n_components=2)
    tsne_vectors = tsne.fit_transform(data)

    plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], s=3, c=labels)
    plt.savefig("cluster.png")
