"""
Author: Tomasz Hachaj, 2024
Department of Applied Computer Science in AGH University of Krakow, Poland.
https://home.agh.edu.pl/~thachaj/

Order of run: 3
"""


import pickle
import numpy as np

def read_me(emb_id, cluster_count, threshold):
    file = open("res/" + str(emb_id) + "_" + str(cluster_count) + "_" + str(threshold) + ".pickle", 'rb')
    data = pickle.load(file)
    return data

def get_data(emb_id, cluster_count, threshold):
    [rs_all_distances, rs_clusters, cluster_id, res_set, [mc, calinski_harabasz_score, davies_bouldin_score, silhouette_score]] = read_me(emb_id, cluster_count, threshold)
    unique_elements_count = len(set(res_set))
    anomalies_count = int(0.005 * unique_elements_count)

    unique_elements = set()
    a = 0
    while len(unique_elements) < anomalies_count:
        unique_elements.add(res_set[a])
        a = a + 1
    return [unique_elements, res_set]

def IOU(unique_elements0, unique_elements1):
    return (len(unique_elements0 & unique_elements1)/len(unique_elements0 | unique_elements1))

def IOU_2(unique_elements0, unique_elements1, res_set0, res_set1):
    return (len((unique_elements0  & set(res_set1)) & (unique_elements1 & set(res_set0)))/len((unique_elements0 & set(res_set1)) | (unique_elements1 & set(res_set0))))



emb_id = 0
threshold = 1.05
cluster_count = 2

emb_id1 = 0
threshold1 = 1.05
cluster_count1 = 2

iou_all = []
iou_2_all = []

count = 0
tt = []

for threshold in np.arange(1.05, 3, 0.2):
    tt.append(round(threshold, 2))
print(tt)
threshold = threshold1 = tt[2]

for cluster_count in range(2, 11, 2):
    tt.append(round(cluster_count, 2))
cluster_count = cluster_count1 = tt[4]
cluster_count = cluster_count1 = tt[1]

import statistics
count = 0

# WYPISANIE WYNIKÓW
import statistics

res_rows = []
for cluster_count in range(2, 11, 2):
    res_row = []
    for threshold in np.arange(1.05, 3, 0.2):
        threshold = round(threshold, 2)
        iou_all = []
        iou_2_all = []
        for emb_id in range(0, 10):
            [unique_elements0, res_set0] = get_data(emb_id, cluster_count, threshold)
            cluster_count1 = cluster_count
            threshold1 = threshold
            for emb_id1 in range(0,10):
                if emb_id1 > emb_id:
                    #threshold1 = round(threshold1, 2)
                    [unique_elements1, res_set1] = get_data(emb_id1, cluster_count1, threshold1)
                    iou = (IOU(unique_elements0, unique_elements1))
                    iou_2 = (IOU_2(unique_elements0, unique_elements1, res_set0, res_set1))

                    my_str = str(emb_id) + "," + str(cluster_count) + "," + str(threshold) + "," \
                             + str(emb_id1) + "," + str(cluster_count1) + "," + str(threshold1) + "," \
                             + str(iou) + "," + str(iou_2)
                    #print(my_str)
                    iou_all.append(iou)
                    iou_2_all.append(iou_2)
        if len(iou_all) > 0:
            # change to iou for the other coefficient
            val = statistics.mean(iou_2_all)
            val = round(val, 2)
            res_row.append(val)
        count = count + 1
    res_rows.append(res_row)

cluster_count = 2
for a in range(len(res_rows)):
    rr = res_rows[a]
    to_screen = str(cluster_count)
    for b in range(len(rr)):
        to_screen = to_screen + " & " + str(rr[b])
    to_screen = to_screen + "\\\\"
    print(to_screen)
    cluster_count = cluster_count + 2

xx = 1

print("")
print("")

res_rows = []
# compate with the best
for cluster_count in range(2, 11, 2):
    res_row = []
    for threshold in np.arange(1.05, 3, 0.2):
        threshold = round(threshold, 2)
        iou_all = []
        iou_2_all = []
        for emb_id in range(0, 10):
            [unique_elements0, res_set0] = get_data(emb_id, cluster_count, threshold)
            cluster_count1 = 2
            threshold1 = 1.65
            emb_id1 = emb_id

            if True:
                if cluster_count == cluster_count1 and threshold == threshold1:
                    xxxxx = 1
                    xxxxx = xxxxx + 1
            #if not(cluster_count == cluster_count1 and threshold == threshold1):
                #threshold1 = round(threshold1, 2)
                [unique_elements1, res_set1] = get_data(emb_id1, cluster_count1, threshold1)
                iou = (IOU(unique_elements0, unique_elements1))
                iou_2 = (IOU_2(unique_elements0, unique_elements1, res_set0, res_set1))

                my_str = str(emb_id) + "," + str(cluster_count) + "," + str(threshold) + "," \
                         + str(emb_id1) + "," + str(cluster_count1) + "," + str(threshold1) + "," \
                         + str(iou) + "," + str(iou_2)
                #print(my_str)
                iou_all.append(iou)
                iou_2_all.append(iou_2)
        if len(iou_all) > 0:
            # to_file = str(count) + ";" + str(cluster_count) + ";" + str(cluster_count1) + ";" + str(threshold) + ";" + str(threshold1) + ";" + ";" + (str(statistics.mean(iou_all)) + ";" + str(statistics.stdev(iou_all)) + ";" + str(statistics.mean(iou_2_all)) + ";" + str(statistics.stdev(iou_2_all)))
            # print(to_file + " " + str(len(iou_all)))
            val = statistics.mean(iou_2_all)
            val = round(val, 2)
            res_row.append(val)
        count = count + 1
    res_rows.append(res_row)


cluster_count = 2
for a in range(len(res_rows)):
    rr = res_rows[a]
    to_screen = str(cluster_count)
    for b in range(len(rr)):
        to_screen = to_screen + " & " + str(rr[b])
    to_screen = to_screen + "\\\\"
    print(to_screen)
    cluster_count = cluster_count + 2
