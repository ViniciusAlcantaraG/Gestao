import pandas as pd

# Load the CSV
df = pd.read_csv('Kohonen_result.csv')

# List your parameter columns here
param_cols = ['Rows', 'Cols', 'LearningRate']

# 1. Cluster size statistics per parameter set
cluster_sizes = (
    df.groupby(param_cols + ['Cluster'])
    .size()
    .reset_index(name='ClusterSize')
    .sort_values(param_cols + ['ClusterSize'], ascending=[True, True, True, False])
)
print("Cluster sizes (top 10):")
print(cluster_sizes.head(10), "\n")

# 2. Class distribution per cluster per parameter set
class_dist = (
    df.groupby(param_cols + ['Cluster', 'TrueClass'])
    .size()
    .reset_index(name='Count')
    .sort_values(param_cols + ['Cluster', 'Count'], ascending=[True, True, True, True, False])
)
print("Class distribution per cluster (top 10):")
print(class_dist.head(10), "\n")

# 3. Purity per cluster per parameter set
def cluster_purity(group):
    return group['TrueClass'].value_counts(normalize=True).iloc[0]

purity_per_cluster = (
    df.groupby(param_cols + ['Cluster'])
    .apply(cluster_purity)
    .reset_index(name='Purity')
    .sort_values(param_cols + ['Purity'], ascending=[True, True, True, False])
)
print("Purity per cluster (top 10):")
print(purity_per_cluster.head(10), "\n")

# 4. Overall average purity per parameter set
avg_purity = (
    purity_per_cluster.groupby(param_cols)['Purity']
    .mean()
    .reset_index(name='AvgClusterPurity')
    .sort_values('AvgClusterPurity', ascending=False)
)
print("Average cluster purity per parameter set:")
print(avg_purity.head(10), "\n")

# 5. Number of clusters per parameter set
n_clusters = (
    df.groupby(param_cols)['Cluster']
    .nunique()
    .reset_index(name='NumClusters')
)
print("Number of clusters per parameter set:")
print(n_clusters.head(10), "\n")