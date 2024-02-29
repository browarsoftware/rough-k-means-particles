"""
Author: Tomasz Hachaj, 2024
Department of Applied Computer Science in AGH University of Krakow, Poland.
https://home.agh.edu.pl/~thachaj/

Order of run: 4
"""

import numpy as np
import matplotlib.pyplot as plt

emb_array = np.load("pca.res/0/emb_array.npy")

x = emb_array[:,0:1]
y = emb_array[:,1:2]

import pickle
import numpy as np
def read_me(emb_id, cluster_count, threshold):
    file = open("res/" + str(emb_id) + "_" + str(cluster_count) + "_" + str(threshold) + ".pickle", 'rb')
    data = pickle.load(file)
    return data

from rkmeans.kmrs import RoughKMeans

no_clusters = 3
wght_lower=0.9
wght_upper=0.1
threshold=1.20
p_param=1
wght=False


data_to_cluster = {}
# Change data format to dictionary, each column is one feature
for a in range(emb_array.shape[1]):
    data_to_cluster['f' + str(a)] = emb_array[:, a].tolist()


np.random.seed(2)
#clstrk = RoughKMeans(data_to_cluster,no_clusters,wght_lower,wght_upper,threshold,p_param,wght)
#clstrk.timing = False
# For plain k-means run with threshold=1
#clstrk = RoughKMeans(data_to_cluster,no_clusters,wght_lower=0.9,wght_upper=0.1,threshold=1.65,p_param=1.,wght=False)
clstrk = RoughKMeans(data_to_cluster,no_clusters,wght_lower,wght_upper,threshold,p_param,wght)
clstrk.timing = False
clstrk.get_rough_clusters()

colors = []
markers = []
markers_list = []
for a in range(emb_array.shape[0]):
    colors.append((0, 0, 0))
    markers.append("o")


boundary = "upper"
clstrk.clusters
rs_clusters = []
cluster_id = []



for i in range(clstrk.max_clusters):
    #rs_clusters += clstrk.clusters[str(i)][boundary]
    for j in clstrk.clusters[str(i)][boundary]:
        cc = colors[j]
        if i == 0:
            colors[j] =  (0.8, cc[1], cc[2])
        if i == 1:
            colors[j] =  (cc[0], 0.8, cc[2])
        if i == 2:
            colors[j] =  (cc[0], cc[1], 0.8)


x1 = x[:,:]
list_xy = []
markers_list = []
for  i in range(clstrk.max_clusters):
    #rs_clusters += clstrk.clusters[str(i)][boundary]
    list_xy.append((x[clstrk.clusters[str(i)][boundary],:], y[clstrk.clusters[str(i)][boundary],:]))
    ll = []
    for j in range(len(clstrk.clusters[str(i)][boundary])):
        if i == 0:
            ll.append('o')
        if i == 1:
            ll.append('+')
        if i == 2:
            ll.append('x')

############################################################

# boundary = "lower" - not used
boundary = "upper"

# Data analysis and visualization
# Get list of all image indexes in the same order as it is in clustering results data structure
rs_clusters = []
cluster_id = []
for i in range(clstrk.max_clusters):
  rs_clusters += clstrk.clusters[str(i)][boundary]

# Compute distances from centroids
rs_all_distances = []
rs_only_upper_distances = []
for i in range(clstrk.max_clusters):
    clt1 = str(i)
    for image_id in clstrk.clusters[clt1][boundary]:
      rs_emb = emb_array[image_id,:]
      rs_emb_center = clstrk.centroids[clt1]
      rs_all_distances.append(np.linalg.norm(rs_emb - rs_emb_center))

      cluster_id.append(i)



# Sort data in descending order and get indexes of sorted data
rs_sort_index = np.flip(np.argsort(rs_all_distances))

unique_elements_count = len(set(rs_clusters))
anomalies_count = int(0.005 * unique_elements_count)
print(anomalies_count)
# Get 25 potential anomalies
#img_number = 25
img_number = anomalies_count
my_ids = []
my_cluster_ids = []
id = 0

while len(my_ids) < img_number:
    idid = rs_clusters[rs_sort_index[id]]
    colors[idid] =  (0, 0, 0)
    markers[idid] = "*"
    if idid not in my_ids:
        my_ids.append(idid)
    id = id + 1

fig, ax = plt.subplots()

markers = ["x","+","o"]
colors = [(0.4,0.4,0.4), (0.6,0.6,0.6), (0.8,0.8,0.8)]
#for i, c in enumerate(np.unique(col)):
for a in range(len(list_xy)):
    plt.scatter(list_xy[a][0],list_xy[a][1],marker=markers[a], color=colors[a])


x_a = x[my_ids,:]
y_x = y[my_ids,:]
plt.scatter(x_a,y_x,marker="*", color=(0,0,0))

plt.title('Clustering results with anomalies detected (k=' + str(no_clusters) + ', eps=' + str(threshold) + ')')
plt.xlabel('PCA dim 1')
plt.ylabel('PCA dim 2')

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0],  color='w', markerfacecolor=(0.4,0.4,0.4), marker='P', label='C1', markersize=10),
                   Line2D([0], [0],  color='w', markerfacecolor=(0.6,0.6,0.6), marker='X', label='C2', markersize=10),
                   Line2D([0], [0],  color='w', markerfacecolor=(0.8,0.8,0.8), marker='o', label='C3', markersize=10),
                   Line2D([0], [0],  color='w', markerfacecolor=(0, 0, 0), marker='*', label='Anomaly', markersize=15)]
ax.legend(handles=legend_elements, loc='upper left')
plt.show()
