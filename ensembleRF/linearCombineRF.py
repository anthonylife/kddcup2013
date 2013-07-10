#!/usr/bin/env python
#encoding=utf8

# Copyright: for KDDCup 2013 track 1, team group work.
# Author: anthonylife
#
# Idea: modify the data distribution to improve the fusion of random forest.
# Procedures: 1.randomly partition the training data into several different parts;
#             2.training random forest on each part of the training data;
#             3.model combination by linear regression.

import csv, json, sys, random
import pickle
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LogisticRegression

def feature_selection(orig_features, filter_idx_set):
    all_feature_num = len(orig_features[0])
    new_features = np.array([[0.0 for i in range(all_feature_num-len(filter_idx_set))] for j in range(len(orig_features))])
    orig_features = np.array(orig_features)

    idx = 0
    for i in range(all_feature_num):
        if i in filter_idx_set:
            continue
        new_features[:,idx] = orig_features[:,i]
        idx += 1

    if idx != (all_feature_num - len(filter_idx_set)):
        print "dimension of generated feature matrix is wrong."
        sys.exit(1)

    new_features = new_features.tolist()

    return new_features


def ct_author_feature_dict(features_conf, features_deleted):
    author_features = defaultdict(dict)
    author_set = set([])

    for entry in features_conf:
        if entry[0] not in author_features:
            author_features[entry[0]]['1'] = []
            author_features[entry[0]]['0'] = []
        author_features[entry[0]]['1'].append(entry[1:])
        author_set.add(entry[0])

    for entry in features_deleted:
        if entry[0] not in author_features:
            author_features[entry[0]]['1'] = []
            author_features[entry[0]]['0'] = []
        author_features[entry[0]]['0'].append(entry[1:])
        author_set.add(entry[0])

    return author_features, author_set


def partition_author(author_set, cv_num):
    cv_author = []
    author_set = list(author_set)
    random.shuffle(author_set)

    author_num  = len(author_set)
    sep_num_set = [0, author_num/cv_num, author_num*2/cv_num, author_num*3/cv_num,\
            author_num*4/cv_num, author_num]
    for i in range(cv_num):
        cv_author.append(author_set[sep_num_set[i]:sep_num_set[i+1]])

    return cv_author


def ct_feature_set(author_features, cv_author):
    ''' Return features with format: "authorid, paperid, features1, feature2, ..."
    '''
    cv_features = []

    for sub_author_set in cv_author:
        features = []
        target = []
        for author in sub_author_set:
            if author not in author_features:
                print 'author key error!'
                sys.exit(1)
            if '1' in author_features[author]:
                for feature in author_features[author]['1']:
                    features.append([author] + feature)
                target += [1 for i in range(len(author_features[author]['1']))]
            if '0' in author_features[author]:
                for feature in author_features[author]['0']:
                    features.append([author] + feature)
                target += [0 for i in range(len(author_features[author]['0']))]
        cv_features.append([features, target])

    return cv_features


def save_model(model, model_path):
    pickle.dump(model, open(model_path, "w"))

def load_model(model_path):
    return pickle.load(open(model_path))

def main():
    #============================================================================
    #--0.Key control variable setting.
    base_learner = False
    model_num = 2
    filter_idx_set = set([11, 14, 64, 16])
    paths = json.loads(open("SETTINGS.json").read())
    #============================================================================

    #============================================================================
    #--1.Preparation for training models: loading data, partition data...
    print("Getting features for deleted papers from the disk files")
    features_conf = [feature for feature in \
            csv.reader(open(paths["trainpos_features"]))]
    features_deleted = [feature for feature in \
            csv.reader(open(paths["trainneg_features"]))]
    author_features, author_set = ct_author_feature_dict(features_conf,\
            features_deleted)
    author_segments = partition_author(author_set, model_num)
    cv_features_set = ct_feature_set(author_features, author_segments)
    #============================================================================

    #============================================================================
    #--2.Model training (Random forest)
    if not base_learner:
        print 'Model Training.'
        for i in range(0, model_num):
            train_features = [feature[2:] for feature in cv_features_set[i][0]]
            train_features = feature_selection(train_features, filter_idx_set)
            print len(train_features[0])
            train_target = cv_features_set[i][1]
            #classifier = RandomForestRegressor(n_estimators=500,
            classifier = RandomForestClassifier(n_estimators=360,
                                                verbose=2,
                                                n_jobs=4,
                                                min_samples_split=10,
                                                random_state=1)
            classifier.fit(train_features, train_target)
            save_model(classifier, paths["model_path"+str(i+1)])
    #============================================================================

    #============================================================================
    #--3.Model combination (Linear Regression).
    print 'Random Forest models combination.'
    item_features = []
    train_target = []
    for i in range(model_num):
        item_features += cv_features_set[i][0]
        train_target += cv_features_set[i][1]
    item_features = [feature[2:] for feature in item_features]
    item_features = feature_selection(item_features, filter_idx_set)
    '''item_features = [feature[2:] for feature in cv_features_set[-1][0]]
    item_features = feature_selection(item_features, filter_idx_set)
    train_target  = cv_features_set[-1][1]
    '''
    train_features = [[] for i in range(len(item_features))]
    for i in range(1, model_num+1):
        classifier = load_model(paths["model_path"+str(i)])
        predictions = classifier.predict_proba(item_features)[:,1]
        #predictions = classifier.predict(item_features)
        train_features = [entry1+[entry2] for entry1, entry2 in \
                zip(train_features, list(predictions))]

    '''regressor = SGDRegressor(loss='squared_loss',
                             n_iter=500,
                             penalty='l2',
                             alpha=0.01,
                             shuffle=True,
                             random_state=1,
                             eta0=0.001,
                             learning_rate='invscaling')
    '''
    regressor = LogisticRegression(penalty='l2',
                       C=0.1)
    regressor.fit(train_features, train_target)
    save_model(regressor, paths['fusion_model_path'])
    #============================================================================


if __name__ == "__main__":
    main()


