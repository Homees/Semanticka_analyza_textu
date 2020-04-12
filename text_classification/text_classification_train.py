# Importing libraries
import codecs
import pickle
import random
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

import stemming


def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=43, replace=False)

    pca = PCA(n_components=2).fit_transform(data[max_items, :].todense())
    tsne = TSNE().fit_transform(PCA(n_components=4).fit_transform(data[max_items, :].todense()))

    idx = np.random.choice(range(pca.shape[0]), size=43, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i * 0.6 / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    plt.show()


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))


print("There are 10 sentences of following three classes on which K-NN classification and K-means clustering is "
      "performed : \n1. Cricket \n2. Artificial Intelligence \n3. Chemistry")
path = "sentences.txt"
labels_color_map = {
    0: '#20b2aa', 1: '#ff7373', 2: '#4700f9', 3: '#005073', 4: '#4d0404',
    5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
}

train_clean_sentences = []
fp = codecs.open(path, 'r', encoding='utf8')
for line in fp:
    line = line.strip()
    cleaned = stemming.clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)

random.shuffle(train_clean_sentences)
vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, max_features=8000, stop_words=stemming.stopWords)
tf_idf_matrix = vectorizer.fit_transform(train_clean_sentences)
X = tf_idf_matrix.todense()

# Clustering the training 30 sentences with K-means technique
k_means = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300)
labels = k_means.fit_predict(tf_idf_matrix)
#pickle.dump(k_means, open("save.pickle", "wb"))
#pickle.dump(vectorizer, open("tfidf.pickle", "wb"))

plot_tsne_pca(tf_idf_matrix, labels)
get_top_keywords(tf_idf_matrix, labels, vectorizer.get_feature_names(), 5)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)
print(pca.explained_variance_ratio_)

# print reduced_data
fig, ax = plt.subplots()
for index, instance in enumerate(reduced_data):
    # print instance, index, labels[index]
    pca_comp_1, pca_comp_2 = reduced_data[index]
    color = labels_color_map[labels[index]]
    ax.scatter(pca_comp_1, pca_comp_2, c=color)
plt.show()

test_sentences = ["Rovnaké príznaky ako ja, mala aj moja tanečná partnerka, s ktorou som chodil na kurz. Dokonca ich "
                  "mala ešte sinejšie. Je diabetička a možno preto sa cítila horšie než ja.",
                  "Nákup bytov za týchto podmienok nepovažuje za podozrivý. To mi ani nenapadlo, že by to niekto "
                  "mohol takto vidieť, uzavrel bývalý vysoký štátny úradník.",
                  "Jedným z fundamentálnych aspektov Pythonu je koncept kolekčných (alebo kontajnerových) typov.",
                  "Prvé známky o osídlení Slovenska pochádzajú z konca paleolitu približne spred 250 tis. rokov."]

test_clean_sentence = []
for test in test_sentences:
    cleaned_test = stemming.clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+", "", cleaned)
    test_clean_sentence.append(cleaned)

test_data = vectorizer.transform(test_clean_sentence)
predicted_labels_k_means = k_means.predict(test_data)

print("\n", test_sentences[0], ":", predicted_labels_k_means[0],
      "\n", test_sentences[1], ":", predicted_labels_k_means[1],
      "\n", test_sentences[2], ":", predicted_labels_k_means[2],
      "\n", test_sentences[3], ":", predicted_labels_k_means[3])
