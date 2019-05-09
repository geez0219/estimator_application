import numpy as np
import cv2
import os
import pickle


def load_data(path):
    # load training data
    x_train = np.empty([200*500, 64, 64, 3], dtype=np.uint8)
    y_train = np.empty([200*500], dtype=np.int32)
    idx_id_list = [0 for _ in range(200)]

    img_i = 0
    class_i = 0
    for dirpath, dirnames, filenames in os.walk(path + "/train"):
        if dirpath.split("/")[-1] == "images":
            idx_id_list[class_i] = dirpath.split("/")[-2]

            for filename in filenames:
                x_train[img_i] = cv2.imread(dirpath + "/" + filename)
                y_train[img_i] = class_i
                img_i += 1
            class_i += 1

    # get the id to name dict
    file = open(path + "/words.txt", "r")
    id_name_dict = {}
    for line in file.read().splitlines():
        key, value = line.split("\t")
        id_name_dict[key] = value

    # get the id to idx dict
    id_idx_dict = {}
    for idx, id in enumerate(idx_id_list):
        id_idx_dict[id] = idx

    idx_name_list = [id_name_dict[id] for id in idx_id_list]

    # load validation data
    x_test = np.empty([200 * 50, 64, 64, 3], dtype=np.uint8)
    y_test = np.empty([200 * 50], dtype=np.int32)

    for dirpath, dirnames, filenames in os.walk(path + "/val/images"):
        for i in range(200*50):
            x_test[i] = cv2.imread("{}/val_{}.JPEG".format(dirpath, i))

    file = open(path+"/val/val_annotations.txt")

    for i, line in enumerate(file.read().splitlines()):
        y_test[i] = int(id_idx_dict[line.split("\t")[1]])

    return (x_train, y_train), (x_test, y_test), idx_name_list

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test), id_name_list = load_data("tiny-imagenet-200")
    pickle.dump(((x_train, y_train), (x_test, y_test)), open("dataset.pkl", "wb"))
    pickle.dump(id_name_list, open("id_name_list.pkl", "wb"))
