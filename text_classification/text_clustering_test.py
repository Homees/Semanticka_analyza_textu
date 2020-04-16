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
if __name__ == '__main__':
    test_sentences = ["minul mesiac splnil podmienk polovičn cenu vedeni účtu strhnut poplatok pln výšk mal platbu kart e pravideln platbu inkas sporeni mám mesiac účtu idú peniah vkladn knižku pýtam strhnut poplatok pln výšk goliat ivet"]

    kmeans = pickle.load(open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/kmeans_model.pickle", "rb"))
    vectorizer = pickle.load(open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/tfidf_model.pickle", "rb"))
    test_data = vectorizer.transform(test_sentences)
    predicted_labels_k_means = kmeans.predict(test_data)
    
    print(predicted_labels_k_means)
    