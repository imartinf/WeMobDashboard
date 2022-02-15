from hdbscan import HDBSCAN

import glob
import pandas
import os
import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def _main_():
    df = pandas.read_csv('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/intervals_07-01-22.csv')
    df = df[df.plate.isin(df.plate.unique()[:10])]
    X = df[["begin_lat", "begin_long"]].to_numpy()

    results = []
    for mcs in [5,10,20,50,100]:
        clustering = HDBSCAN(min_cluster_size=mcs, min_samples=50, metric="haversine")
        clustering.fit(X)
        labels = clustering.labels_
        nb_labels = labels.max()
        print('Min cluster size: ', mcs, 'Number of labels: ', nb_labels)
        # print('Labels: ', set(labels))
        if nb_labels > 1:
            sil_noise = silhouette_score(X, labels, metric="haversine",
                random_state=1234)
            print('Silhouette w/ noise: ', sil_noise)
            X_ = [x for x, l in zip(X, labels) if l != -1]
            lables = [l for l in labels if l != -1]
            sil_no_noise = silhouette_score(X_, lables, metric="haversine",
                random_state=1234)
            print('Silhouette w/o noise: ', sil_no_noise)
            nb_noise = (len(labels) - len(lables)) / len(labels)
            print('Number of noise samples: ', nb_noise)
            results.append([mcs, nb_labels, sil_noise, sil_no_noise, nb_noise])
        print('--------------------------------')
    df = pandas.DataFrame(results, columns=[
        'min_cluster_size', 'nb_clusters', 'silhouette w/noise', 'silhouette w/o noise', 'noise'])
    df.to_csv('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/out/cluster_hdbscan_exp2_50ms.csv')


if __name__ == "__main__":
    _main_()