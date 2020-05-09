# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:52:51 2020

@author: 973065
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import string
import codecs
import pickle
import re
import random

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

from numpy import unicode
from scipy import sparse
from time import time
from itertools import cycle
from itertools import product
%matplotlib inline

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn import metrics

from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

WORD_PATTERN = re.compile(r"^\w+$", re.UNICODE)

stop_words = {"a", "aby", "aj", "ak", "ako", "ale", "alebo", "and", "ani", "áno", "asi", "až", "bez", "bude", "budem",
             "budeš", "budeme", "budete", "budú", "by", "bol", "bola", "boli", "bolo", "byť", "cez", "čo", "či",
             "ďalší", "ďalšia", "ďalšie", "dnes", "do", "ho", "ešte", "for", "i", "ja", "je", "jeho", "jej", "ich",
             "iba", "iné", "iný", "som", "si", "sme", "sú", "k", "kam", "každý", "každá", "každé", "každí", "kde",
             "keď", "kto", "ktorá", "ktoré", "ktorou", "ktorý", "ktorí", "ku", "lebo", "len", "ma", "mať", "má", "máte",
             "medzi", "mi", "mna", "mne", "mnou", "musieť", "môcť", "môj", "môže", "my", "na", "nad", "nám", "náš",
             "naši", "nie", "nech", "než", "nič", "niektorý", "nové", "nový", "nová", "noví", "o", "od", "odo", "of",
             "on", "ona", "ono", "oni", "ony", "po", "pod", "podľa", "pokiaľ", "potom", "práve", "pre", "prečo",
             "preto", "pretože", "prvý", "prvá", "prvé", "prví", "pred", "predo", "pri", "pýta", "s", "sa", "so",
             "svoje", "svoj", "svojich", "svojím", "svojími", "ta", "tak", "takže", "táto", "teda", "te", "tě", "ten",
             "tento", "the", "tieto", "tým", "týmto", "tiež", "to", "toto", "toho", "tohoto", "tom", "tomto", "tomuto",
             "tu", "tú", "túto", "tvoj", "ty", "tvojími", "už", "v", "vám", "váš", "vaše", "vo", "viac", "však",
             "všetok", "vy", "z", "za", "zo", "že", "buď", "ju", "menej", "moja", "moje", "späť", "ste", "tá", "tam",
             "dobrý", "deň", "ďakujem", "pozdravom", "dakujems", "dakujem", "prosim", 'no', 'this', 
             'jan', 'has', 'been', 'messag', 'wagner', 'wer', 'iss', 'discovered', 'analyzed', 'tremboš',
             'ivet', 'goliat', 'as', 'kask', 'krisk', 'account', 'you', 'address', 'amount',
             'mám', 'vás', 'chcem', 'dás', 'začk'}
exclude = set(string.punctuation)


#%%
def stem(word, aggressive=False):
    if not isinstance(word, unicode):
        word = word.decode("utf8")

    if not WORD_PATTERN.match(word):
        return word

    if not word.islower() and not word.istitle() and not word.isupper():
        print("warning: skipping word with mixed case: {}".format(word), file=sys.stderr)
        return word

    # all our pattern matching is done in lowercase
    s = word.lower()
    s = _remove_case(s)
    s = _remove_possessives(s)

    if aggressive:
        s = _remove_comparative(s)
        s = _remove_diminutive(s)
        s = _remove_augmentative(s)
        s = _remove_derivational(s)

    if word.isupper():
        return s.upper()
    if word.istitle():
        return s.title()
    return s


def _remove_case(word):
    if len(word) > 7 and word.endswith("atoch"):
        return word[:-5]
    if len(word) > 6:
        if word.endswith("aťom"):
            return _palatalise(word[:-3])
    if len(word) > 5:
        if word[-3:] in ("och", "ich", "ích", "ého", "ami", "emi", "ému",
                         "ete", "eti", "iho", "ího", "ími", "imu", "aťa"):
            return _palatalise(word[:-2])
        if word[-3:] in ("ách", "ata", "aty", "ých", "ami",
                         "ové", "ovi", "ými"):
            return word[:-3]
    if len(word) > 4:
        if word.endswith("om"):
            return _palatalise(word[:-1])
        if word[-2:] in ("es", "ém", "ím"):
            return _palatalise(word[:-2])
        if word[-2:] in ("úm", "at", "ám", "os", "us", "ým", "mi", "ou", "ej"):
            return word[:-2]
    if len(word) > 3:
        if word[-1] in "eií":
            return _palatalise(word)
        if word[-1] in "úyaoáéý":
            return word[:-1]
    return word


def _remove_possessives(word):
    if len(word) > 5:
        if word.endswith("ov"):
            return word[:-2]
        if word.endswith("in"):
            return _palatalise(word[:-1])
    return word


def _remove_comparative(word):
    if len(word) > 5:
        if word[-3:] in ("ejš", "ějš"):
            return _palatalise(word[:-2])
    return word


def _remove_diminutive(word):
    if len(word) > 7 and word.endswith("oušok"):
        return word[:-5]
    if len(word) > 6:
        if word[-4:] in ("ečok", "éčok", "ičok", "íčok", "enok", "énok",
                         "inok", "ínok"):
            return _palatalise(word[:-3])
        if word[-4:] in ("áčok", "ačok", "očok", "učok", "anok", "onok",
                         "unok", "ánok"):
            return _palatalise(word[:-4])
    if len(word) > 5:
        if word[-3:] in ("ečk", "éčk", "ičk", "íčk", "enk", "énk",
                         "ink", "ínk"):
            return _palatalise(word[:-3])
        if word[-3:] in ("áčk", "ačk", "očk", "učk", "ank", "onk",
                         "unk", "átk", "ánk", "ušk"):
            return word[:-3]
    if len(word) > 4:
        if word[-2:] in ("ek", "ék", "ík", "ik"):
            return _palatalise(word[:-1])
        if word[-2:] in ("ák", "ak", "ok", "uk"):
            return word[:-1]
    if len(word) > 3 and word[-1] == "k":
        return word[:-1]
    return word


def _remove_augmentative(word):
    if len(word) > 6 and word.endswith("ajzn"):
        return word[:-4]
    if len(word) > 5 and word[-3:] in ("izn", "isk"):
        return _palatalise(word[:-2])
    if len(word) > 4 and word.endswith("ák"):
        return word[:-2]
    return word


def _remove_derivational(word):
    if len(word) > 8 and word.endswith("obinec"):
        return word[:-6]
    if len(word) > 7:
        if word.endswith("ionár"):
            return _palatalise(word[:-4])
        if word[-5:] in ("ovisk", "ovstv", "ovišt", "ovník"):
            return word[:-5]
    if len(word) > 6:
        if word[-4:] in ("ások", "nosť", "teln", "ovec", "ovík",
                         "ovtv", "ovin", "štin"):
            return word[:-4]
        if word[-4:] in ("enic", "inec", "itel"):
            return _palatalise(word[:-3])
    if len(word) > 5:
        if word.endswith("árn"):
            return word[:-3]
        if word[-3:] in ("enk", "ián", "ist", "isk", "išt", "itb", "írn"):
            return _palatalise(word[:-2])
        if word[-3:] in ("och", "ost", "ovn", "oun", "out", "ouš",
                         "ušk", "kyn", "čan", "kář", "néř", "ník",
                         "ctv", "stv"):
            return word[:-3]
    if len(word) > 4:
        if word[-2:] in ("áč", "ač", "án", "an", "ár", "ar", "ás", "as"):
            return word[:-2]
        if word[-2:] in ("ec", "en", "ér", "ír", "ic", "in", "ín",
                         "it", "iv"):
            return _palatalise(word[:-1])
        if word[-2:] in ("ob", "ot", "ov", "oň", "ul", "yn", "čk", "čn",
                         "dl", "nk", "tv", "tk", "vk"):
            return word[:-2]
    if len(word) > 3 and word[-1] in "cčklnt":
        return word[:-1]
    return word


def _palatalise(word):
    if word[-2:] in ("ci", "ce", "či", "če"):
        return word[:-2] + "k"

    if word[-2:] in ("zi", "ze", "ži", "že"):
        return word[:-2] + "h"

    if word[-3:] in ("čte", "čti", "čtí"):
        return word[:-3] + "ck"

    if word[-3:] in ("šte", "šti", "ští"):
        return word[:-3] + "sk"
    return word[:-1]


# Cleaning the text sentences so that punctuation marks, stop words &amp; digits are removed
def clean(doc):
    stop_free = " ".join([j for j in doc.lower().split() if j not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(stem(word, aggressive=False) for word in punc_free.split())
    processed = re.sub(r"\d+", "", normalized)
    y = processed.split()
    return y


def map_to_list(emails, key):
    results = []

    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])

    return results



def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df



def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = np.asarray(X[grp_ids])
    else:
        D = np.asarray(X)
        
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=25):
    dfs = []
    labels = np.unique(y)
    
    for label in labels:
        ids = np.where(y==label) 
        feats_df = top_mean_feats(X, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(20, 10), facecolor="w")
    x = np.arange(len(dfs[0]))

    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("cluster = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score, align='center', color='#7530FF')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.features)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
    
    
def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0
    # Set initial number of features
    n_components = 0
    
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
            
    # Return the number of components
    return n_components
    

#%%
if __name__ == '__main__':
    path = "/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/emails.txt"
    
    train_clean_sentences = []
    fp = codecs.open(path, 'r', encoding='utf8')
    for line in fp:
        line = line.strip()
        cleaned = clean(line)
        cleaned = ' '.join(cleaned)
        train_clean_sentences.append(cleaned)

    dataset = []
    random.shuffle(train_clean_sentences)
    dataset.append(train_clean_sentences[:2000])
    dataset.append(train_clean_sentences[2000:4000])
    dataset.append(train_clean_sentences[4000:6000])
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000)
    
    sparse_matrices = []
    for data in dataset:
        original_X = vectorizer.fit_transform(data)
        feature_names = vectorizer.get_feature_names()
        
        sparse_matrix = original_X.toarray().sum(axis=0)
        vocabulary_dict = {}
        for key in feature_names:
            feature_index = vectorizer.vocabulary_.get(key)
            vocabulary_dict.update({key : sparse_matrix[feature_index]})
        
        vocabulary_dict = sorted(vocabulary_dict.items(), key=lambda x: x[1], reverse=True)
        new_vocabulary = []
        for val in vocabulary_dict:
            new_vocabulary.append(val[0])
        #print(vocabulary_dict[:50])
         
        vectorizer = TfidfVectorizer(vocabulary=new_vocabulary[:200], stop_words=stop_words, max_features=10000)
        original_X = vectorizer.fit_transform(data)
        sparse_matrices.append(original_X)
        print(original_X.shape)
    
    print("Performing dimensionality reduction using LSA")
    
    output_matrices = []
    for matrix in sparse_matrices:
        """
        svd = TruncatedSVD(n_components=matrix.shape[1] - 1).fit(matrix)
        explained_variance = svd.explained_variance_ratio_
        
        dim = [50, 60, 70, 80]
        perplexity = [10, 20, 30, 40, 50, 60]
        lr = [20, 50, 100, 200, 500, 1000]
        init = [500, 750, 1000, 2000, 3000]
        cartesian_product = product(dim, perplexity, lr, init)
        
        for product in cartesian_product:
            #n = select_n_components(explained_variance, product[0])
            print('Number of dimentions %s: ' % product[0])
        """
        t0 = time()
        svd = TruncatedSVD(n_components=50)
        #tsne = TSNE(n_components=2, perplexity=product[1], learning_rate=product[2], n_iter=product[3])
        tsne = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=3000)
        scaler = MinMaxScaler()
        lsa = make_pipeline(svd, scaler)
    
        #X = svd.fit_transform(matrix)
        #X = scaler.fit_transform(X)
        X = lsa.fit_transform(matrix)
        X = tsne.fit_transform(X)
        output_matrices.append(X)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        #print("perplexity: %s, learning rate: %s, n_init: %s, time: (%.3g sec)" % (product[1], product[2], product[3], time() - t0))
        ax.scatter(X[:, 0], X[:, 1], c='b', cmap=plt.cm.Spectral)
        #ax.set_title("perplexity: %s, learning rate: %s, n_init: %s, time: (%.3g sec)" % (product[1], product[2], product[3], time() - t0))
        ax.grid()
        #fig.savefig('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/tsne_plots_1/'+str(product[0])+'_'+str(product[1])+'_'+str(product[2])+'_'+str(product[3]), dpi=250)
        plt.show()
            
    """
            explained_variance = svd.explained_variance_ratio_.sum()
            print("done in %fs" % (time() - t0))
            print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
            print()
            
        break
    
    """
    np.save('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/2d_matrix_1.npy', output_matrices[0])
    np.save('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/2d_matrix_2.npy', output_matrices[1])
    np.save('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/2d_matrix_3.npy', output_matrices[2])
    sparse.save_npz('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/sparse_matrix_1.npz', sparse_matrices[0])
    sparse.save_npz('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/sparse_matrix_2.npz', sparse_matrices[1])
    sparse.save_npz('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/sparse_matrix_3.npz', sparse_matrices[2])
    pickle.dump(vectorizer, open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/tfidf.pickle", "wb"))
    