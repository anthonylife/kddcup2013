#!/usr/bin/env python
#encoding=utf8

# Copyright: KDDCup 2013 team work
# Author: anthonylife
#
# Feature selection: utilizing scikit-learn package, i.e., sklearn.feature_selection.RFECV

import csv, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

def main():
    print("Loading paths")
    paths = json.loads(open("SETTINGS.json").read())

    print("Getting features for deleted papers from the disk files")
    features_conf = [feature for feature in csv.reader(open(paths["trainpos_features"]))]
    features_deleted = [feature for feature in csv.reader(open(paths["trainneg_features"]))]

    features = np.array([map(float, x[2:]) for x in features_deleted + features_conf])
    target = np.array([0 for x in range(len(features_deleted))] + [1 for x in range(len(features_conf))])

    '''classifier = RandomForestClassifier(n_estimators=360,
                                        verbose=2,
                                        n_jobs=4,
                                        min_samples_split=10,
                                        random_state=1)
    '''
    classifier = SVR(kernel="linear")
    print("Start feature selection")
    selector = RFECV(classifier, step=1, cv=5)
    print features.shape
    selector = selector.fit(features, target)

    print("Ouput feature selection results")
    print selector.support_
    print selector.ranking_

    writer = csv.writer(open(paths["selection_result"], "w"))
    writer.writerow(selector.support_.tolist())
    writer.writerow(selector.ranking_.tolist())

if __name__=="__main__":
    main()


