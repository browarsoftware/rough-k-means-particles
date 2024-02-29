"""
Author: Tomasz Hachaj, 2024
Department of Applied Computer Science in AGH University of Krakow, Poland.
https://home.agh.edu.pl/~thachaj/
Data source: data/credo_data.zip

Order of run: 5
"""

# Preapare data
import numpy as np
# Open file with embedding
emb_array = np.load("pca.res/6/emb_array.npy")
emb_array = emb_array[:,0:62]
print(emb_array.shape)

# Open file with dataset files names
#my_file = open("./data/image_files_list.txt", "r")
my_file = open("image_files_list_6.txt", "r")

data = my_file.read()
data_into_list = data.split("\n")
my_file.close()
# Import code for rough k-means
# Original file can be found here: https://github.com/geofizx/rough-clustering
from rkmeans.kmrs import RoughKMeans
from gap_statistic import OptimalK

data_to_cluster = {}
# Change data format to dictionary, each column is one feature
for a in range(emb_array.shape[1]):
  data_to_cluster['f' + str(a)] = emb_array[:,a].tolist()

# Compute RoughKMeans algorithm
np.random.seed(0)
#clstrk = RoughKMeans(data_to_cluster,8,wght_lower=0.9,wght_upper=0.1,threshold=2.5,p_param=1.,wght=False)
clstrk = RoughKMeans(data_to_cluster,3,wght_lower=0.9,wght_upper=0.1,threshold=1.65,p_param=1.,wght=False)
# For plain k-means run with threshold=1
#clstrk = RoughKMeans(data_to_cluster,4,wght_lower=0.9,wght_upper=0.1,threshold=1,p_param=1.,wght=False)
clstrk.get_rough_clusters()

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
      xxxxx = clstrk.clusters[clt1]["lower"]
      if image_id not in xxxxx:
          rs_only_upper_distances.append(np.linalg.norm(rs_emb - rs_emb_center))


mc = sum(rs_only_upper_distances) / len(rs_only_upper_distances)
print("**************************")
print(str(mc) + " (" + str(len(rs_only_upper_distances)) + ")")
print("**************************")
print(rs_only_upper_distances[0:20])

# Sort data in descending order and get indexes of sorted data
rs_sort_index = np.flip(np.argsort(rs_all_distances))

# Get 25 potential anomalies
#img_number = 25
img_number = 64
my_ids = []
my_cluster_ids = []
id = 0
while len(my_ids) < img_number:
  id_help = data_into_list[rs_clusters[rs_sort_index[id]]]
  if id_help not in my_ids:
    my_ids.append(data_into_list[rs_clusters[rs_sort_index[id]]])
    my_cluster_ids.append(cluster_id[rs_sort_index[id]])
  id = id + 1

# Plot results
import matplotlib.pyplot as plt
import cv2

fig = plt.figure(figsize=(8, 8))
columns = 8
rows = int(len(my_ids) / 8) + 1

columns2 = 11
rows2 = int(len(my_ids) / columns2) + 1
ret_img = np.zeros((128 * rows2, 128 * columns2, 3))

xx = 0
yy = 0
for aaa in range(len(my_ids)):
    img_help = cv2.imread('d:/data/data_small/' + my_ids[aaa])
    aaa1 = aaa + 1

    ret_img[128 * yy : 128 * (yy + 1), 128 * xx : 128 * (xx + 1), :] = cv2.resize(img_help, (128, 128))
    xx = xx + 1

    if xx > columns2 - 1:
        xx = 0
        yy = yy + 1

    fig.add_subplot(rows, columns, aaa1)
    plt.axis("off")  # turns off axes
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.axis("tight")  # gets rid of white border
    plt.imshow(img_help)

print(my_cluster_ids)
print(my_ids)
plt.axis('off')
plt.show()