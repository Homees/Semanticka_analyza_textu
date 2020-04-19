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
    test_sentences = ["dobrý deň minulý mesiac som splnil podmienky na polovičnú cenu za vedenie účtu a bol mi strhnutý poplatok v plnej výške mal som platbu kartou cez e pravidelnú platbu inkaso a sporenie mám tiež každý mesiac mi z účtu idu peniaze na vkladnú knížku pytam sa prečo mi bol strhnutý poplatok v plnej výške dakujem  goliatová iveta",
                      " dobrý deň  chcela by som sa vas opytať na jednu otázkuzaložila som si šikovné sporenie a pani ktorá mi ho uzatvárala mi povedala že všetko si budem moc kontrolovať a vidite cez internet bankingále zatiaľ od prvého prevodu a už aj druhého som na mojom konte nevidela  za pomoc vopred dakujem  andrea lommele  goliatová iveta",
                      "príkaz nie je možné zadať z dôvodu neprípustného dátumu splatnosti alebo obmedzenej funkčnosti systému v prípade otázok kontaktujte sporotel tel this message has been analyzed and no issues were discovered goliatová iveta",
                      "dobrý deň  pri zmene hesla mi v druhom kroku neprichádza autorizačný kod na mobilný telefón tj zmenu hesla nemôžem dokončiť  dakujem za preverenie s pozdravom jk goliatová iveta"]

    kmeans = pickle.load(open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/kmeans_model.pickle", "rb"))
    vectorizer = pickle.load(open("/u00/au973065/git_repo/Semanticka_analyza_textu/text_classification/save/tfidf_model.pickle", "rb"))
    test_data = vectorizer.transform(test_sentences)
    predicted_labels_k_means = kmeans.predict(test_data)
    
    print(predicted_labels_k_means)
    