"""
Author: Tomasz Hachaj, 2024
Department of Applied Computer Science in AGH University of Krakow, Poland.
https://home.agh.edu.pl/~thachaj/
Data source: data/credo_data_align.zip

Order of run: 1
"""

import random
image_files_list = "image_files_list_all.txt"
# path to dataset
path = 'd:\\dane\\rough_set\\data_align\\'

with open("image_files_list_all.txt", 'r') as fp:
    all_files = fp.readlines()
for a in range(len(all_files)):
    all_files[a] = all_files[a].strip()


print(all_files[0:10])
random.shuffle(all_files)
all_files_len = len(all_files)
count_help = int(all_files_len / 10)
ranges = []
for a in range(10):
    if a == 0:
        ranges.append(all_files[count_help:all_files_len])
    else:
        ranges.append(all_files[0:(a * count_help)])
        if a < 10 - 1:
            ranges[a] += all_files[((a + 1) * count_help):all_files_len]

    with open("image_files_list_" + str(a) + ".txt", 'w') as fp:
        for ff in ranges[a]:
            fp.write(ff + "\n")
