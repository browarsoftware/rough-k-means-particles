"""
Author: Tomasz Hachaj, 2024
Department of Applied Computer Science in AGH University of Krakow, Poland.
https://home.agh.edu.pl/~thachaj/
Data source: data/credo_data_align.zip

Order of run: 2
"""

import cv2
import numpy as np
import os

#scale image
def scale(np_i):
    np1 = np.copy(np_i)
    np2 = (np1 - np.min(np1)) / np.ptp(np1)
    return np2


#scale and reshape image for visualization
def scale_and_reshape(np_i, mf, old_shape):
    np1 = np.copy(np_i)
    if mf is None:
        np2 = np1.reshape(old_shape, order='F')
    else:
        np2 = (np1 + mf).reshape(old_shape, order='F')
    np2 = scale(np2)
    return np2

#encode using eigenfaces (steganography)
def encode(data_to_code, carrier_img_i, v, mean_face, message_offset):
    carrier_img = np.copy(carrier_img_i)
    old_shape = carrier_img.shape
    img_flat = carrier_img.flatten('F')
    img_flat -= mean_face
    # generate eigenfaces from carier
    result = np.matmul(v.transpose(), img_flat)
    result_message = result
    ssss = len(result_message)
    result_message[50:len(result_message)] = 0
    # store message in features vector
    #result_message[message_offset:(message_offset + data_to_code.shape[0])] = data_to_code
    # reconstruct carrier image
    reconstruct_message = np.matmul(v, result_message)
    image_to_code2 = scale_and_reshape(reconstruct_message, mean_face, old_shape)
    return image_to_code2

#encode using eigenfaces (steganography)
def embedding(carrier_img_i, v, mean_face):
    carrier_img = np.copy(carrier_img_i)
    old_shape = carrier_img.shape
    img_flat = carrier_img.flatten('F')
    img_flat -= mean_face
    # generate eigenfaces from carier
    result = np.matmul(v.transpose(), img_flat)
    #result_message = result
    #ssss = len(result_message)
    #result_message[50:len(result_message)] = 0
    return result

path = 'd:/dane/rough_set/data_align/'

if not os.path.exists("pca.res"):
    os.makedirs("pca.res")
for dir_name in range(10):
    if not os.path.exists("pca.res/" + str(dir_name)):
        os.makedirs("pca.res/" + str(dir_name))

    path_to_results = "pca.res/" + str(dir_name)

    image_files_list = "image_files_list_" + str(dir_name) + ".txt"

    files = []
    flip_mat = False

    with open(image_files_list, 'r') as fp:
        files = fp.readlines()
    for a in range(len(files)):
        files[a] = files[a].strip()
    how_many_images = len(files)


    img = cv2.imread(path + files[0], cv2.IMREAD_GRAYSCALE)
    old_shape = img.shape
    img_flat = img.flatten('F')

    T = np.zeros((img_flat.shape[0], len(files)))
    for i in range(len(files)):
        if i % 1000 == 0:
            print("\tLoading " + str(i) + " of " + str(len(files)))
        img_help = cv2.imread(path + files[i], cv2.IMREAD_GRAYSCALE)
        T[:,i] = img_help.flatten('F') / 255


    print('Calculating mean face')
    mean_face = T.mean(axis = 1)

    for i in range(len(files)):
        T[:,i] -= mean_face


    print('Calculating covariance')
    if flip_mat:
        C = np.matmul(T.transpose(), T)
    else:
        C = np.matmul(T, T.transpose())

    C = C / how_many_images

    print('Calculating eigenfaces')
    from scipy.linalg import eigh
    w, v = eigh(C)

    if flip_mat:
        v_correct = np.matmul(T, v)
    else:
        v_correct = v

    sort_indices = w.argsort()[::-1]
    w = w[sort_indices]  # puttin the evalues in that order
    v_correct = v_correct[:, sort_indices]


    norms = np.linalg.norm(v_correct, axis=0)# find the norm of each eigenvector
    v_correct = v_correct / norms

    #save results
    np.save(path_to_results + "//T_st_", T)
    np.save(path_to_results + "//v_st_", v_correct)
    np.save(path_to_results + "//w_st_", w)
    np.save(path_to_results + "//mean_face_st_", mean_face)
    np.save(path_to_results + "//norms_st_", norms)
    np.save(path_to_results + "//old_shape_st_", np.asarray(old_shape))



    all_embedding = []
    all_files = []

    with open(image_files_list, 'r') as fp:
        all_files = fp.readlines()

    print('Calculating embedding')
    import os
    # r=root, d=directories, f = files
    for file in all_files:
        #files.append(os.path.join(r, file))
        full_path = os.path.join(path, str.strip(file))
        img_help = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        off = 0
        embed = embedding(img_help / 255, v_correct, mean_face)
        all_embedding.append(embed)
        #all_files.append(file)
        #img_help = cv2.imread(files[i + offset], cv2.IMREAD_GRAYSCALE)


    emb_array = np.zeros((len(all_embedding), len(all_embedding[0])))
    a = 0

    for a in range(len(all_embedding)):
        emb_array[a, :] = all_embedding[a]

    np.save(path_to_results + "/emb_array", emb_array)
