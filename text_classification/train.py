# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:32:03 2020

@author: 973065
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import string
import codecs
import pickle
import re

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LogNorm
from sklearn import mixture

from numpy import unicode
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram
from time import time
from itertools import cycle
%matplotlib inline

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn import metrics

from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

#%%
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    
    plt.grid()
    plt.show()
    
    
def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=2000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    idx = np.random.choice(range(pca.shape[0]), size=200, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, (ax1, ax2) = plt.subplots(2, figsize=(10, 20))
    
    ax1.scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax1.set_title('PCA Cluster Plot')
    
    ax2.scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax2.set_title('TSNE Cluster Plot')
    
    plt.grid()
    plt.show()
    
    
def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(', '.join([labels[t] for t in np.argsort(r)[-n_terms:]]))    
    
  
#%%
if __name__ == '__main__':
    X = np.load('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/2d_matrix_1.npy')
    original_X = sparse.load_npz('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/sparse_matrix_1.npz')
    vectorizer = pickle.load(open('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/tfidf.pickle', 'rb'))
    
    find_optimal_clusters(X, 30)
    find_optimal_clusters(original_X, 30)
    
    true_k = 26
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(11, 40))
    
    t0 = time()
    #print('Clustering sparse data with %s' % kmeans)
    #kmeans = MiniBatchKMeans(n_clusters=true_k, init_size=1024, batch_size=2048, random_state=20)
    #kmeans = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000)
    kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    #kmeans.fit(X)
    clusters = kmeans.fit_predict(X)
    
    #plot_tsne_pca(original_X, clusters)
    #get_top_keywords(original_X, clusters, vectorizer.get_feature_names(), 10)
    centers = kmeans.cluster_centers_
    
    #pickle.dump(kmeans, open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/kmeans_1.pickle", "wb"))
    
    ax1.scatter(X[:, 0], X[:, 1], c=clusters, label=clusters, s=50, cmap='viridis')
    ax1.scatter(centers[:, 0], centers[:, 1], c='r', marker='^', s=100, alpha=0.8);
    ax1.set_title('Kmeans clustering with number of clusters: %d done in %0.3fs' % (true_k, time() - t0))
    ax1.grid()
    ax1.legend()
    #plt.show()
    
    print('K-means clustering done in %0.3fs' % (time() - t0))
    print()
    
    # #############################################################################
    # Compute clustering with MeanShift
    
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.03, n_samples=500)
    
    t0 = time()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    clusters = ms.fit_predict(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    #pickle.dump(ms, open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/meanshift_1.pickle", "wb"))
    #get_top_keywords(original_X, clusters, vectorizer.get_feature_names(), 10)
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    # #############################################################################
    # Plot result
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        ax2.plot(X[my_members, 0], X[my_members, 1], col + '.')
        ax2.plot(cluster_center[0], cluster_center[1], 'o', label=k, markerfacecolor=col, markeredgecolor='k', markersize=14)
    
    ax2.set_title('Mean shift clustering with number of estimated clusters: %d done in %0.3fs' % (n_clusters_, time() - t0))
    ax2.grid()
    ax2.legend()
    #plt.show()
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Mean shift clustering done in %0.3fs' % (time() - t0))
    print()
    
    # #############################################################################
    # Compute DBSCAN
    t0 = time()
    db = DBSCAN(eps=3.3, min_samples=11).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    #pickle.dump(db, open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/dbscan_1.pickle", "wb"))
    #get_top_keywords(original_X, labels, vectorizer.get_feature_names(), 10)
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    
    # #############################################################################
    # Plot result
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        ax3.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), label=k, markeredgecolor='k', markersize=14)
    
        xy = X[class_member_mask & ~core_samples_mask]
        ax3.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    
    ax3.set_title('DBSCAN clustering with estimated number of clusters: %d done in %0.3fs' % (n_clusters_, time() - t0))
    ax3.grid()
    ax3.legend()
    #plt.show()
    
    print('DBSCAN clustering done in %0.3fs' % (time() - t0))
    print()
    
    # #############################################################################
    # Compute Affinity Propagation
    t0 = time()
    af = AffinityPropagation(damping=0.8).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    
    n_clusters_ = len(cluster_centers_indices)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    
    # #############################################################################
    # Plot result    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        ax4.plot(X[class_members, 0], X[class_members, 1], col + '.')
        ax4.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        
        for x in X[class_members]:
            ax4.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    
    ax4.set_title('Affinity Propagation with estimated number of clusters: %d done in %0.3fs' % (n_clusters_, time() - t0))
    ax4.grid()
    ax4.legend()
    #plt.show()
    
    print('Affinity Propagation done in: %0.3fs' % (time() - t0))
    print()
    
    fig.savefig('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/data_clustering_plots_1.png', dpi=200)
    plt.show()
