''' Slovak stemmer
Adapted from the Czech stemmer with the following copyright
    Copyright © 2010 Luís Gomes <luismsgomes@gmail.com>.
It was also inspired by sumy:
    https://github.com/miso-belica/sumy/blob/dev/sumy/nlp/stemmers/czech.py
Ported from the Java implementation available at:
    http://members.unine.ch/jacques.savoy/clef/index.html
'''
from __future__ import division, print_function, unicode_literals

import string
import sys
import re

from numpy import unicode

WORD_PATTERN = re.compile(r"^\w+$", re.UNICODE)


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
             "všetok", "vy", "z", "za", "zo", "že", "buď", "ju", "menej", "moja", "moje", "späť", "ste", "tá", "tam"}
exclude = set(string.punctuation)
