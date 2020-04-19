# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:32:03 2020

@author: 973065
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import codecs
import pickle
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
%matplotlib inline

from numpy import unicode
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")

WORD_PATTERN = re.compile(r"^\w+$", re.UNICODE)

stopWords = {"a", "aby", "aj", "ak", "ako", "ale", "alebo", "and", "ani", "áno", "asi", "až", "bez", "bude", "budem",
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
             "dobrý", "deň", "ďakujem", "pozdravom"}
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
    stop_free = " ".join([j for j in doc.lower().split() if j not in stopWords])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(stem(word, aggressive=False) for word in punc_free.split())
    processed = re.sub(r"\d+", "", normalized)
    y = processed.split()
    return y

  
#%%
if __name__ == '__main__':
    path = "/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/data/lm_diacritized_emails.txt"
    
    train_clean_sentences = []
    fp = codecs.open(path, 'r', encoding='utf8')
    for line in fp:
        line = line.strip()
        cleaned = clean(line)
        cleaned = ' '.join(cleaned)
        train_clean_sentences.append(cleaned)
        
    print(train_clean_sentences)

    vectorizer = TfidfVectorizer(stop_words = stopWords, max_features = 1000)
    vectorizer = vectorizer.fit(train_clean_sentences[:14000])
    X = vectorizer.transform(train_clean_sentences[:14000])
    word_features = vectorizer.get_feature_names()
    print(X.shape)
    print(word_features[:50])
    
    #wcss = []
    #for i in range(1,21):
    #    kmeans = KMeans(n_clusters=i, init='k-means++',max_iter=300, n_init=10, random_state=0)
    #    kmeans.fit(X)
    #    wcss.append(kmeans.inertia_)
        
    #fig, ax = plt.subplots(figsize=(12, 10))
    #ax.plot(range(1,21), wcss)
    #plt.title('The Elbow Method')
    #plt.xlabel('Number of clusters')
    #plt.ylabel('WCSS')
    #fig.savefig('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/elbow_old.png')
    #plt.show()
    
    kmeans = KMeans(n_clusters = 9, n_init = 20, n_jobs = 1)
    kmeans.fit(X)
    # Finally, we look at 8 the clusters generated by k-means.
    common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(word_features[word] for word in centroid))
        
    labels_color_map = {
        0: '#fc9d03', 1: '#fceb03', 2: '#cefc03', 3: '#6ffc03', 4: '#03fcad',
        5: '#03fce7', 6: '#03adfc', 7: '#033dfc', 8: '#9d03fc', 9: '#f803fc',
        10: '#fc03b5', 11: '#fcdb03', 12: '#41fc03', 13: '#03fca9', 14: '#03a9fc',
        15: '#fcfc03'
    }
    
    pickle.dump(kmeans, open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/kmeans_model_9c_14k.pickle", "wb"))
    pickle.dump(vectorizer, open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/tfidf_model_9c_14k.pickle", "wb"))
    
    labels = kmeans.predict(X)
    pca = PCA(n_components=2).fit(X.todense())
    flatten_data = pca.transform(X.todense())
    print(pca.explained_variance_ratio_)
    centers2D = pca.transform(kmeans.cluster_centers_)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    for index, instance in enumerate(flatten_data):
        pca_comp_1, pca_comp_2 = flatten_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color, alpha=0.6)

    ax.grid(True)
    plt.hold(True)
    plt.scatter(centers2D[:,0], centers2D[:,1], marker='^', s=200, c='#ab0000')
        
    plt.title('Text clusters for 14000 emails')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    fig.savefig('/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/clusters_9c_14k.png')
    plt.show()
    