import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from sklearn.cluster import KMeans

def init_k_means(num_clusters, init, max_iters, random_state=69):
    return KMeans(n_clusters=num_clusters, init=init, max_iter=max_iters, random_state=random_state)

def run_k_means(model, input_array):
    model.fit(input_array)
    return model.labels_, model.cluster_centers_

def save_labels_to_heatmap(labels):
    cmap = plt.get_cmap('viridis')
    # Normalize the labels
    norm = mcolors.Normalize(vmin=labels.min(), vmax=labels.max())
    # Convert the labels to a colormap
    heatmap_colors = cmap(norm(labels))
    colored_npy = (heatmap_colors[:, :, :3] * 255).astype(np.uint8)
    return colored_npy

if __name__ == "__main__":
    ARRAY_PATH = r"pixel_vector_data_train_files_transformed.npy"
    NUM_CLUSTERS = 100
    INIT = "k-means++"
    MAX_ITERS = 500

    X = np.load(ARRAY_PATH)
    X = X[:12851*250]
    print(X.shape)
    print("File loaded")
    model_k_means = init_k_means(NUM_CLUSTERS, INIT, MAX_ITERS)
    labels, clusters= run_k_means(model_k_means, X)
    print(clusters.shape)
    print(labels.shape)
    
    y = save_labels_to_heatmap(labels.reshape(12851,250))
    cv.imwrite("for_ppt.png", y)
