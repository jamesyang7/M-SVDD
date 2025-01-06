from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tab_colors = plt.get_cmap('tab10').colors

def plot_ellipsoid(mu, sigma_inv, X_train, X_test, mahalanobis_distances=[1.4, 2, 3, 4], file_name='ellipsoid_plot.png'):
    """
    Plots the ellipsoid based on Mahalanobis distance, as well as X_train and X_test.
    
    Parameters:
    - mu: Mean vector.
    - sigma_inv: Inverse covariance matrix.
    - X_train: Training data.
    - X_test: Test data.
    - mahalanobis_distances: List of Mahalanobis distances to plot the ellipsoids for.
    - file_name: Name of the file to save the plot as a PNG.
    """
    
    # Step 1: Define a grid of points in the space of the first two features (for visualization)
    x_min, x_max = np.min(np.vstack((X_train, X_test))[:, 0]-0.05), np.max(np.vstack((X_train, X_test))[:, 0])
    y_min, y_max = np.min(np.vstack((X_train, X_test))[:, 1]), np.max(np.vstack((X_train, X_test))[:, 1])
    
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[x.ravel(), y.ravel()]

    # Step 2: Calculate Mahalanobis distance for each point in the grid
    mahalanobis_grid = []
    for point in grid_points:
        diff = point - mu[:2]
        mahalanobis_grid.append(np.sqrt(np.dot(np.dot(diff.T, sigma_inv[:2, :2]), diff)))

    mahalanobis_grid = np.array(mahalanobis_grid).reshape(x.shape)

    # Step 3: Plot the scatter plot of training and testing data
    plt.figure(figsize=(10, 7))

    # Default orange and blue colors for train and test
    plt.scatter(X_test[:, 0], X_test[:, 1], alpha=0.3, color=tab_colors[1])
    plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.2, color=tab_colors[0])


    # Step 4: Plot the ellipsoid contours for specified Mahalanobis distances
    for d in mahalanobis_distances:
        plt.contour(x, y, mahalanobis_grid, levels=[d], colors='r', alpha=0.4, linewidths=2)

    # Step 5: Plot the mean (center) of the ellipsoid
    plt.scatter(mu[0], mu[1], color='r', marker='*', s=150)

    # Step 6: Automatically set axis limits based on the max Mahalanobis distance
    max_distance = np.max(mahalanobis_distances) /35
    plt.xlim(mu[0] - max_distance, mu[0] + max_distance)
    plt.ylim(mu[1] - max_distance, mu[1] + max_distance)
    plt.axis('off')
    if file_name!='':
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=300)


def plot_tsne(total_z_normal,total_z_anomaly):
    X_train = np.array(total_z_normal)
    X_test = np.array(total_z_anomaly)

    X_combined = np.vstack((X_train, X_test))
    tsne = TSNE(n_components=2, random_state=0, n_iter=500)  # Adjust n_iter if necessary
    X_tsne = tsne.fit_transform(X_combined)

    X_train_tsne = X_tsne[:len(X_train)]
    X_test_tsne = X_tsne[len(X_train):]

    plt.figure(figsize=(10, 7))
    plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], alpha=0.4)
    plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], alpha=0.5)
    plt.axis('off')
    plt.show()