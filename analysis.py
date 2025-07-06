import pandas as pd

# Load the CSV
df = pd.read_csv('DBSCAN_result.csv')

# 1. Noise Rate per parameter set
noise_rate = (
    df.groupby(['Radius', 'MinPoints'])
    .apply(lambda x: (x['Cluster'] == -1).mean())
    .reset_index(name='NoiseRate')
)

# 2. Cluster Purity per parameter set (excluding noise)
def purity(group):
    # For each cluster, find the most common true class
    clusters = group[group['Cluster'] != -1].groupby('Cluster')
    purities = []
    for _, cgroup in clusters:
        most_common = cgroup['TrueClass'].value_counts(normalize=True).iloc[0]
        purities.append(most_common)
    return sum(purities) / len(purities) if purities else 0

purity_df = (
    df.groupby(['Radius', 'MinPoints'])
    .apply(purity)
    .reset_index(name='AvgClusterPurity')
)

# 3. Combine for best parameter selection
summary = pd.merge(noise_rate, purity_df, on=['Radius', 'MinPoints'])
# Sort by highest purity and lowest noise
summary = summary.sort_values(['AvgClusterPurity', 'NoiseRate'], ascending=[False, True])

print(summary)