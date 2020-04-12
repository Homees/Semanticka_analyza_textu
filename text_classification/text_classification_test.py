# Importing libraries
import pickle
import re
from collections import Counter

import stemming

# Predicting it on test data : Testing Phase
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

kmeans = pickle.load(open("save.pickle", "rb"))
vectorizer = pickle.load(open("tfidf.pickle", "rb"))
test_data = vectorizer.transform(test_clean_sentence)
predicted_labels_k_means = kmeans.predict(test_data)

print("\nBelow 3 sentences will be predicted against the learned nieghbourhood and learned clusters:\n1. ",
      test_sentences[0], "\n2. ", test_sentences[1], "\n3. ", test_sentences[2])

print("\n-------------------------------PREDICTIONS BY K-Means--------------------------------------")
print("\nIndex of Slovakia cluster : ", Counter(kmeans.labels_[0:8]).most_common(1)[0][0])
print("Index of Python cluster : ", Counter(kmeans.labels_[8:16]).most_common(1)[0][0])
print("Index of Economy cluster : ", Counter(kmeans.labels_[16:25]).most_common(1)[0][0])
print("Index of news cluster : ", Counter(kmeans.labels_[25:33]).most_common(1)[0][0])

print("\n", test_sentences[0], ":", predicted_labels_k_means[0],
      "\n", test_sentences[1], ":", predicted_labels_k_means[1],
      "\n", test_sentences[2], ":", predicted_labels_k_means[2],
      "\n", test_sentences[3], ":", predicted_labels_k_means[3])
