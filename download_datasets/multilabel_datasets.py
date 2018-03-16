import numpy as np
import pandas as pd
import os
import requests
import zipfile
import sys


class multilabeled_data(object):
    def __init__(self, dataset_name, X, y, X_names, y_names):
        self._dataset_name = dataset_name
        self._features = X
        self._labels = y
        self._n_samples, self._n_labels = self._labels.shape
        self._n_features = self._features.shape[1]
        self._feature_names = X_names
        self._label_names = y_names

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def name(self):
        return self._dataset_name

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def n_labels(self):
        return self._n_labels

    @property
    def n_features(self):
        return self._n_features

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def label_names(self):
        return self._label_names

def _exists_dataset(dataset_name, data_home_dir):
    save_dir = data_home_dir + dataset_name
    labels_path = save_dir + "/" + dataset_name + "_labels.csv"
    features_path = save_dir + "/" + dataset_name + "_features.csv"
    # make data_home_dir if not exist.
    if not os.path.exists(data_home_dir):
        os.mkdir(data_home_dir)
        os.mkdir(save_dir)
        return False

    # make save_dir if not exist.
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        return False

    if os.path.exists(labels_path) and os.path.exists(features_path):
        return True
    else:
        return False

def _download(save_dir, download_url):
    filename = download_url.split('/')[-1]
    save_file = save_dir + "/" + filename
    r = requests.get(download_url, stream=True)
    with open(save_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()
        return save_file
    # if you cannot open file
    return False

def keel2csv(dat_file, features_path, labels_path):
    attributes = []
    read_data = False
    features = []
    labels = []
    with open(dat_file, "r") as f:
        for row in f:
            elems = row.strip().split(" ")
            if read_data:
                values = np.array(elems[0].split(","))
                feature = values[b_attributes]
                label = values[np.logical_not(b_attributes)]
                features.append(feature)
                labels.append(label)
            elif elems[0] == "@attribute":
                attributes.append(elems[1])
            elif elems[0] == "@inputs":
                input_ = row.strip().replace(',', '').split(" ")[1:]
                b_attributes = np.array([True if att in input_ else False for att in attributes])
            elif elems[0] == "@outputs":
                output_ = row.strip().replace(',', '').split(" ")[1:]
            elif elems[0] == "@data":
                read_data = True
        features = np.array(features)
        labels = np.array(labels)
        _save_csv(features_path, features, ",".join(input_))
        _save_csv(labels_path, labels, ",".join(output_))
        return True
    return False

def _save_csv(path, contents, header):
    with open(path, 'w') as f:
        f.write(header)
        for content in contents:
            f.write('\n')
            f.write(','.join(content))
        return True
    return False

def load_file(save_dir, download_url, delete_zip):
    save_file = _download(save_dir, download_url)
    if save_file:
        print('{} is downloaded.'.format(save_file))
    else:
        raise FileNotFoundError

    # unzip
    zfile = zipfile.ZipFile(save_file)
    zfile.extractall(save_dir)
    print('unzip {}.'.format(save_file))

    # delete zip file
    if delete_zip:
        os.remove(save_file)
        print('{} is deleted.'.format(save_file))



def load_scene(data_home_dir="./data_home/", return_features_labels=False, delete_zip=False, delete_dat=False):
    dataset_name = "scene"
    save_dir = data_home_dir + dataset_name
    labels_path = save_dir + "/" + dataset_name + "_labels.csv"
    features_path = save_dir + "/" + dataset_name + "_features.csv"
    # in the case where you have already download scene datasets to local
    if not _exists_dataset(dataset_name, data_home_dir):
        # download zip file
        download_url = "http://sci2s.ugr.es/keel/dataset/data/multilabel/scene.zip"

        # load zip file -> unzip -> dat file
        load_file(save_dir, download_url, delete_zip)

        # keel format -> csv format
        dat_file = save_dir + "/" + dataset_name + ".dat"
        keel2csv(dat_file, features_path, labels_path)

        # delete dat file
        if delete_dat:
            os.remove(dat_file)
            print('{} is deleted.'.format(dat_file))

    # read labels data
    labels = pd.read_csv(labels_path)
    label_names = list(labels.columns)
    labels = np.array(labels)

    # read features data
    features = pd.read_csv(features_path)
    feature_names = list(features.columns)
    features = np.array(features)

    if return_features_labels:
        return features, labels
    else:
        return multilabeled_data(dataset_name, features, labels, feature_names, label_names)

def load_emotions(data_home_dir="./data_home/", return_features_labels=False, delete_zip=False, delete_dat=False):
    dataset_name = "emotions"
    save_dir = data_home_dir + dataset_name
    labels_path = save_dir + "/" + dataset_name + "_labels.csv"
    features_path = save_dir + "/" + dataset_name + "_features.csv"
    # in the case where you have already download scene datasets to local
    if not _exists_dataset(dataset_name, data_home_dir):
        # download zip file
        download_url = "http://sci2s.ugr.es/keel/dataset/data/multilabel/emotions.zip"

        # load zip file -> unzip -> dat file
        load_file(save_dir, download_url, delete_zip)

        # keel format -> csv format
        dat_file = save_dir + "/" + dataset_name + ".dat"
        keel2csv(dat_file, features_path, labels_path)

        # delete dat file
        if delete_dat:
            os.remove(dat_file)
            print('{} is deleted.'.format(dat_file))

    # read labels data
    labels = pd.read_csv(labels_path)
    label_names = list(labels.columns)
    labels = np.array(labels)

    # read features data
    features = pd.read_csv(features_path)
    feature_names = list(features.columns)
    features = np.array(features)

    if return_features_labels:
        return features, labels
    else:
        return multilabeled_data(dataset_name, features, labels, feature_names, label_names)

def load_yeast(data_home_dir="./data_home/", return_features_labels=False, delete_zip=False, delete_dat=False):
    dataset_name = "yeast"
    save_dir = data_home_dir + dataset_name
    labels_path = save_dir + "/" + dataset_name + "_labels.csv"
    features_path = save_dir + "/" + dataset_name + "_features.csv"
    # in the case where you have already download scene datasets to local
    if not _exists_dataset(dataset_name, data_home_dir):
        # download zip file
        download_url = "http://sci2s.ugr.es/keel/dataset/data/multilabel/yeast.zip"

        # load zip file -> unzip -> dat file
        load_file(save_dir, download_url, delete_zip)

        # keel format -> csv format
        dat_file = save_dir + "/" + dataset_name + ".dat"
        keel2csv(dat_file, features_path, labels_path)

        # delete dat file
        if delete_dat:
            os.remove(dat_file)
            print('{} is deleted.'.format(dat_file))

    # read labels data
    labels = pd.read_csv(labels_path)
    label_names = list(labels.columns)
    labels = np.array(labels)

    # read features data
    features = pd.read_csv(features_path)
    feature_names = list(features.columns)
    features = np.array(features)

    if return_features_labels:
        return features, labels
    else:
        return multilabeled_data(dataset_name, features, labels, feature_names, label_names)

def load_enron(data_home_dir="./data_home/", return_features_labels=False, delete_zip=False, delete_dat=False):
    dataset_name = "enron"
    save_dir = data_home_dir + dataset_name
    labels_path = save_dir + "/" + dataset_name + "_labels.csv"
    features_path = save_dir + "/" + dataset_name + "_features.csv"
    # in the case where you have already download scene datasets to local
    if not _exists_dataset(dataset_name, data_home_dir):
        # download zip file
        download_url = "http://sci2s.ugr.es/keel/dataset/data/multilabel/enron.zip"

        # load zip file -> unzip -> dat file
        load_file(save_dir, download_url, delete_zip)

        # keel format -> csv format
        dat_file = save_dir + "/" + dataset_name + ".dat"
        keel2csv(dat_file, features_path, labels_path)

        # delete dat file
        if delete_dat:
            os.remove(dat_file)
            print('{} is deleted.'.format(dat_file))

    # read labels data
    labels = pd.read_csv(labels_path)
    label_names = list(labels.columns)
    labels = np.array(labels)

    # read features data
    features = pd.read_csv(features_path)
    feature_names = list(features.columns)
    features = np.array(features)

    if return_features_labels:
        return features, labels
    else:
        return multilabeled_data(dataset_name, features, labels, feature_names, label_names)

def load_medical(data_home_dir="./data_home/", return_features_labels=False, delete_zip=False, delete_dat=False):
    dataset_name = "medical"
    save_dir = data_home_dir + dataset_name
    labels_path = save_dir + "/" + dataset_name + "_labels.csv"
    features_path = save_dir + "/" + dataset_name + "_features.csv"
    # in the case where you have already download scene datasets to local
    if not _exists_dataset(dataset_name, data_home_dir):
        # download zip file
        download_url = "http://sci2s.ugr.es/keel/dataset/data/multilabel/medical.zip"

        # load zip file -> unzip -> dat file
        load_file(save_dir, download_url, delete_zip)

        # keel format -> csv format
        dat_file = save_dir + "/" + dataset_name + ".dat"
        keel2csv(dat_file, features_path, labels_path)

        # delete dat file
        if delete_dat:
            os.remove(dat_file)
            print('{} is deleted.'.format(dat_file))

    # read labels data
    labels = pd.read_csv(labels_path)
    label_names = list(labels.columns)
    labels = np.array(labels)

    # read features data
    features = pd.read_csv(features_path)
    feature_names = list(features.columns)
    features = np.array(features)

    if return_features_labels:
        return features, labels
    else:
        return multilabeled_data(dataset_name, features, labels, feature_names, label_names)

def main():
    # test code
    data = load_medical()
    print(data.label_names)
    print(data.feature_names)
    print(data.features)
    print(data.labels)
    print(data.features.shape)
    print(data.labels.shape)
    print(data.n_samples)
    print(data.n_labels)
    print(data.n_features)

if __name__ == '__main__':
    main()
