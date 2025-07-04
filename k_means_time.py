import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Plotting function to visualize the time taken for k-means clustering with different k values
def plot_kMeansTime(k_values, times):

    plt.figure(figsize=(10, 6))
    plt.scatter(k_values, times, color='blue', label='Data Points')

    # Linear regression
    k_array = np.array(k_values)
    t_array = np.array(times)
    coeffs = np.polyfit(k_array, t_array, 1)  # Linear fit (degree 1)
    linear_fit = np.poly1d(coeffs)
    plt.plot(k_array, linear_fit(k_array), color='red', linestyle='--', label='Linear Regression')

    plt.title('K-Means Clustering Time vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Time (microseconds)')
    plt.grid(True)
    plt.xticks(k_values)
    plt.legend()
    plt.show()
    plt.savefig("k_means_time.png")

# Example usage
if __name__ == "__main__":
    # Use the first columns of K_means_times.csv as k_values and the second column as times

    data = pd.read_csv('K_means_times.csv')
    k_values = data.iloc[:, 0].tolist()  # First column as k values
    times = data.iloc[:, 1].tolist()    # Second column as times
    plot_kMeansTime(k_values, times)