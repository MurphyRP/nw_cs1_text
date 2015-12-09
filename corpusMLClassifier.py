from __future__ import division, print_function
import json
import math
from pandas import DataFrame
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron


# handful of helper functions to build tf, idf, weight (no custom weights) and word counts in DSI

def tf(doclines, doclength):
    if doclines > 0:
        return round(doclines / doclength, 5)
    else:
        return 0

def idf(tfval, corpuslength):
    if tfval > 0:
        return round(math.log(corpuslength/tfval), 5)
    else:
        return 0


def tdf_idf_weight(tfval, idfval):
    weight = tfval * idfval
    if weight:
        return weight
    else:
        return 0

def get_class_index(doc_class, class_index):
    return class_index.get(doc_class)

def get_word_count(dsi_string):
    word_count = len(re.split(r'\w+', (dsi_string.replace("'", ""))))
    return word_count

#------------------------------------------
with open('cs1_terms.txt', 'rU') as data_file:
    lines = set()
    for line in data_file:
        lines.add(line.rstrip())


word_list = []
word_weight = []
class_list_Y = []
class_index = {}
doc_classes = set()
#------------------------------------------
#------------------------------------------

with open('cs1_DSI_JSON.txt') as data_file:
    data = json.load(data_file)
    print('file loaded')

    corpus_words = 0
    for c in range(0, len(data)):
        corpus_words += get_word_count(data[c]["extracted"])
        doc_classes.add(data[c]['class'])

    #-------------------------
    c_index = 0
    for doc_class in doc_classes:
        class_index[doc_class] = c_index
        c_index += 1

    for i in range(0, len(data)):
        term = []
        #-------------------------
        key_terms = {}

        for line in lines:

            tfval = tf(data[i]["extracted"].count(line), get_word_count(data[i]["extracted"]))
            idfval = idf(tfval, corpus_words)
            term.append(tdf_idf_weight(tfval, idfval))

        word_weight.append(term)
        class_list_Y.append(get_class_index(data[i]["class"], class_index))

# ---------------------------------------------------
data_frame = DataFrame(np.array(word_weight), columns=lines)

# ---------------------------------------------------


trainX, testX, trainY, testY = train_test_split(data_frame, class_list_Y, test_size=0.4)
# ---------------------------------------------------

clf = MultinomialNB()
clf.fit(trainX, trainY)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
clf_pred = clf.predict(testX)

# ---------------------------------------------------

nlf = Perceptron()
nlf.fit(trainX, trainY)
Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=500, shuffle=True,
           verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False)
nlf_pred = nlf.predict(testX)

# ---------------------------------------------------

rlf = RandomForestClassifier()
rlf.fit(trainX, trainY)
RandomForestClassifier(n_estimators=1000)
rlf_pred = rlf.predict(testX)
# ---------------------------------------------------

c_score = metrics.accuracy_score(testY, clf_pred)
print("MultinomialNB Score")
print("accuracy:   %0.3f" % c_score)

cm = confusion_matrix(testY, clf_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
# ----------------------------------------------------
n_score = metrics.accuracy_score(testY, nlf_pred)
print("Perceptron Score")
print("accuracy:   %0.3f" % n_score)

# Compute confusion matrix
cm = confusion_matrix(testY, nlf_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)


r_score = metrics.accuracy_score(testY, rlf_pred)
print("RandomForestClassifier Score")
print("accuracy:   %0.3f" % r_score)
# ---------------------------------------------------
print(rlf_pred)

# Compute confusion matrix
cm = confusion_matrix(testY, rlf_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)

#--------------------------------------------------------------

rlf_p = RandomForestClassifier()
rlf_p.fit(trainX, trainY)
RandomForestClassifier(n_estimators=1000)
rlf_pred_p = rlf.predict_proba(testX)

#TODO: implement method accepting a string and returning a term vector which can be passed to the classifier for probability
# work

print(rlf_pred_p)

