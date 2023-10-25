import numpy as np

num_vectors = 1000
num_clusters = 3
num_steps = 100

vector_values = []
for i in range(num_vectors):
  if np.random.random() > 0.5:
    vector_values.append([np.random.normal(0.5, 0.6),
                          np.random.normal(0.3, 0.9)])
  else:
    vector_values.append([np.random.normal(2.5, 0.4),
                         np.random.normal(0.8, 0.5)])    
vectors = np.array(vector_values)

indices = np.arange(vectors.shape[0])
np.random.shuffle(indices)
centroids = vectors[indices[:num_clusters]]

for step in range(num_steps):
  diff = vectors - centroids
  squared_diff = np.square(diff)
  distances = np.sum(squared_diff, axis=2)
  assignments = np.argmin(distances, axis=0)
  means = []
  for c in range(num_clusters):
      cluster_points = vectors[assignments == c]
      mean = np.mean(cluster_points, axis=0)
      means.append(mean)
  centroids = np.vstack(means)

print(centroids)