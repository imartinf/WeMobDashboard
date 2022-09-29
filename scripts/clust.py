import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../../')

import glob
import pandas
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import random


EARTH_RADIUS=6371000

def _main_():
    df = pandas.read_csv('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/intervals_1month_9trucks.csv')
    # df = df[df.plate.isin(df.plate.unique()[:10])]
    X = df[["begin_lat", "begin_long"]].to_numpy()

	# neigh = NearestNeighbors(n_neighbors=2, metric="cosine")
	# nbrs = neigh.fit(X)
	# distances, indices = nbrs.kneighbors(X)
	# distances = numpy.sort(distances, axis=0)
	# distances = distances[:,1]
	# plt.plot(distances)
	# plt.xlabel('Sample index')
	# plt.ylabel('Distance')
	# plt.show()

    results = []
    for e in [i/EARTH_RADIUS for i in [50, 100, 200, 500, 750, 1000]]:
        for ms in [2, 4, 6, 10]:
            clustering = DBSCAN(eps=e, min_samples=ms, metric="haversine")
            labels = clustering.fit_predict(np.radians(X))
            nb_labels = len(set(labels)) - 1
            print('Epsilon: ', e, 'Number of labels: ', nb_labels)
            print('Min samples: ', ms)
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
                results.append([e,e*EARTH_RADIUS, ms, nb_labels, sil_noise, sil_no_noise, nb_noise])
        print('--------------------------------')
    df = pandas.DataFrame(results, columns=[
        'epsilon', 'epsilon (m)', 'min_samples', 'nb_clusters', 'silhouette w/noise', 'silhouette w/o noise', 'noise'])
    df.to_csv('cluster_dbscan_exp2.csv')
    # clustering = DBSCAN(eps=0.05, metric="cosine")
    # labels = clustering.fit_predict(X)

    # x_pca = PCA(3).fit_transform(X)

    # fig = pyplot.figure()
    # ax = Axes3D(fig)
    # colors = cm.rainbow(numpy.linspace(0, 1, len(labels)))
    # for x, l, c in zip(x_pca, labels, colors):
    # 	ax.scatter(x[0], x[1], x[2], color=c, label=l)
    # pyplot.legend()
    # pyplot.show()





if __name__ == "__main__":
    _main_()