"""
Author: Tomasz Hachaj, 2024
Department of Applied Computer Science in AGH University of Krakow, Poland.
https://home.agh.edu.pl/~thachaj/
Data source: data/credo_data_align.zip

Order of run: 3
"""

from rkmeans.kmrs import RoughKMeans




def PerformRoughKMeans(data_to_cluster,emb_array, no_clusters=4,wght_lower=0.9,wght_upper=0.1,threshold=1.25,p_param=1.,wght=False):
    np.random.seed(2)
    clstrk = RoughKMeans(data_to_cluster,no_clusters,wght_lower,wght_upper,threshold,p_param,wght)
    clstrk.timing = False
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

    lower_ids = []
    lower_clusters_id = []

    for i in range(clstrk.max_clusters):
        clt1 = str(i)
        for j in clstrk.clusters[clt1]["lower"]:
            lower_clusters_id.append(i)
        lower_ids = lower_ids + clstrk.clusters[clt1]["lower"]
    #lower_ids_set = set(lower_ids)
    emb_array_ids = emb_array[lower_ids,]
    emb_array_ids_arr = np.array(emb_array_ids)
    lower_clusters_id_arr = np.array(lower_clusters_id)

    X = emb_array_ids_arr
    labels = lower_clusters_id_arr
    return [X, labels, clstrk, mc, rs_all_distances, rs_clusters, cluster_id]

from sklearn import metrics
import pickle
import numpy as np
import os

if not os.path.exists("res"):
    os.makedirs("res")

# Open file with embedding
for emb_id in range(0,10):
    emb_array = np.load("pca.res/" + str(emb_id) + "/emb_array.npy")
    emb_array = emb_array[:,0:62]
    print(emb_array.shape)

    # Open file with dataset files names
    #my_file = open("./data/image_files_list.txt", "r")
    my_file = open("image_files_list_" + str(emb_id) + ".txt", "r")

    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    # Import code for rough k-means
    # Original file can be found here: https://github.com/geofizx/rough-clustering
    #from gap_statistic import OptimalK

    data_to_cluster = {}
    # Change data format to dictionary, each column is one feature
    for a in range(emb_array.shape[1]):
      data_to_cluster['f' + str(a)] = emb_array[:,a].tolist()



    for cluster_count in range(2, 11, 2):
        for threshold in np.arange(1.05, 3, 0.2):
            threshold = round(threshold, 2)
            print(str(emb_id) + ", " + str(cluster_count) + ", " + str(threshold))
            [X, labels, clstrk, mc, rs_all_distances, rs_clusters, cluster_id] = PerformRoughKMeans(data_to_cluster,emb_array, no_clusters=cluster_count,wght_lower=0.9,wght_upper=0.1,threshold=threshold,p_param=1.,wght=False)

            calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)
            davies_bouldin_score = metrics.davies_bouldin_score(X, labels)
            silhouette_score = metrics.silhouette_score(X, labels)

            res_print = str(calinski_harabasz_score) + "," + str(davies_bouldin_score) + "," + str(silhouette_score) + "," + str(mc)
            #print(res_print)

            # Sort data in descending order and get indexes of sorted data
            rs_sort_index = np.flip(np.argsort(rs_all_distances))

            res_set = []
            for a in range(len(rs_sort_index)):
                res_set.append(data_into_list[rs_clusters[rs_sort_index[a]]])

            data_to_save = [rs_all_distances, rs_clusters, cluster_id, res_set, [mc, calinski_harabasz_score, davies_bouldin_score, silhouette_score]]

            # open a file, where you ant to store the data
            file = open("res/" + str(emb_id) + "_" + str(cluster_count) + "_" + str(threshold) + ".pickle", 'wb')
            pickle.dump(data_to_save, file)
            file.close()
