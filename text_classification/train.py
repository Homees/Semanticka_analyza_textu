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
        
    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    ax.grid()
    f.savefig('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/elbow_plot_2d', dpi=250)
    
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
    
    
def get_top_keywords(data, clusters, labels, n_terms, algorithm):
    print('Geting top keywords for algorithm: %s' % algorithm)
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i, r in df.iterrows():
        print('\nCluster: {}'.format(i))
        #print('\nWords in r: {}'.format(r))
        print(', '.join([labels[t] for t in np.argsort(r)[-n_terms:]]))    
    
  
#%%
if __name__ == '__main__':
    X = np.load('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/2d_matrix_1.npy')
    original_X = sparse.load_npz('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/sparse_matrix_1.npz')
    vectorizer = pickle.load(open('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/tfidf.pickle', 'rb'))
    
    #find_optimal_clusters(X, 40)
    #find_optimal_clusters(original_X, 40)
    
    true_k = 28
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12, 40))
    
    t0 = time()
    kmeans = KMeans(n_clusters=true_k, init='k-means++', n_jobs=-1)
    clusters = kmeans.fit_predict(X)
    labels = kmeans.labels_
    
    plot_tsne_pca(original_X, clusters)
    get_top_keywords(original_X, clusters, vectorizer.get_feature_names(), 8, kmeans)
    centers = kmeans.cluster_centers_
    
    labels_unique = np.unique(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(labels_unique))]
    
    for k, col in zip(labels_unique, colors):
        members = labels == k
        cluster_center = centers[k]
        ax1.scatter(X[members, 0], X[members, 1], c=col, label=k, s=50, cmap='viridis')
        ax1.scatter(cluster_center[0], cluster_center[1], c='r', marker='^', s=100, alpha=0.8)
        
    ax1.set_title('Kmeans clustering with number of clusters: %d done in %0.3fs' % (true_k, time() - t0))
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='medium')
    ax1.grid()
    #plt.show()
    
    print('K-means clustering done in %0.3fs' % (time() - t0))
    print()
    
    # #############################################################################
    # Compute clustering with MeanShift
    
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.03, n_samples=1000)
    
    t0 = time()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    clusters = ms.fit_predict(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    get_top_keywords(original_X, clusters, vectorizer.get_feature_names(), 8, "Mean Shift")
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    # #############################################################################
    # Plot result
    #colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(labels_unique))]
    
    for k, col in zip(labels_unique, colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        ax2.plot(X[my_members, 0], X[my_members, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
        ax2.plot(cluster_center[0], cluster_center[1], 'o', label=k, markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    
    ax2.set_title('Mean shift clustering with number of estimated clusters: %d done in %0.3fs' % (n_clusters_, time() - t0))
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax2.grid()
    #plt.show()
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Mean shift clustering done in %0.3fs' % (time() - t0))
    print()
    
    # #############################################################################
    # Compute DBSCAN
    t0 = time()
    db = DBSCAN(eps=3.45, min_samples=15, n_jobs=-1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    get_top_keywords(original_X, labels, vectorizer.get_feature_names(), 8, db)
    
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
        ax3.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14, label=k)
        
        xy = X[class_member_mask & ~core_samples_mask]
        ax3.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    
    ax3.set_title('DBSCAN clustering with estimated number of clusters: %d done in %0.3fs' % (n_clusters_, time() - t0))
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax3.grid()
    #plt.show()
    
    print('DBSCAN clustering done in %0.3fs' % (time() - t0))
    print()
    
    # #############################################################################
    # Compute Affinity Propagation
    t0 = time()
    af = AffinityPropagation(damping=0.8, preference=-3500, verbose=True)
    clusters = af.fit_predict(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    
    n_clusters_ = len(cluster_centers_indices)
    get_top_keywords(original_X, clusters, vectorizer.get_feature_names(), 8, af)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    
    # #############################################################################
    # Plot result    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        ax4.plot(X[class_members, 0], X[class_members, 1], col + '.')
        ax4.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14, label=k)
        
        for x in X[class_members]:
            ax4.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    
    ax4.set_title('Affinity Propagation with estimated number of clusters: %d done in %0.3fs' % (n_clusters_, time() - t0))
    ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax4.grid()
    #plt.show()
    
    print('Affinity Propagation done in: %0.3fs' % (time() - t0))
    print()
    
    fig.savefig('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/data_clustering_plots_1.png', dpi=200)
    plt.show()
