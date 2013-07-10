#!/usr/bin/env python
#encoding=utf8

# Copyright: for KDDCup 2013 track 1, team group work.
# Author: anthonylife (WayneZhang)
#
# Idea of model combination: utilize random forest as a weak learner and
#   integrate all the base learners with adaboost framework.
#
# Modifications to traditinoal Adaboost to adapt to probability output of
#   Random Forest:
#   1.error rate of misclassification; (target - prob)
#   2.data point importance distribution; (convert [0,1] to [-1,1])
#   3.weight of linear combination. (normalization)

import csv, json, sys
sys.path.append("../")
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from util import feature_selection, get_confirmed_paper, get_train_confirmed
from util import reorder, save_model, load_model
from evaluation.MAPUtil import calcMAP


def main():
    #============================================================================
    #--1.Preparation for adaboost learning framework with RF as base learner.
    maxnum_iters = 10
    filter_idx_set = set([11, 14, 64, 16])
    paths = json.loads(open("SETTINGS.json").read())

    print("Getting features for deleted papers from the disk files")
    features_conf = [feature for feature in \
            csv.reader(open(paths["trainpos_features"]))]
    features_deleted = [feature for feature in \
            csv.reader(open(paths["trainneg_features"]))]
    train_author_confirmed = get_train_confirmed([pair[:2] for pair in features_conf])
    train_features = [map(float, x[2:]) for x in features_deleted + features_conf]
    train_author_paper_ids = [x[:2] for x in features_deleted + features_conf]
    train_features = feature_selection(train_features, filter_idx_set)
    train_target = [0 for x in range(len(features_deleted))] + \
            [1 for x in range(len(features_conf))]
    train_labels = np.array([-1 for x in range(len(features_deleted))] + \
            [1 for x in range(len(features_conf))])

    features_valid = [feature for feature in \
            csv.reader(open(paths["vali_features"]))]
    test_features = [map(float, x[2:]) for x in features_valid]
    test_features = feature_selection(test_features, filter_idx_set)
    test_author_confirmed = get_confirmed_paper(paths["vali_solution"])
    test_author_paper_ids = [x[:2] for x in features_valid]
    #============================================================================

    #============================================================================
    #--2.Start of adaboost learning framework
    # initialization importance distribution of data points
    trdata_importance = np.array([1.0/len(train_target) for i in\
            range(len(train_target))])
    model_weights = np.array([0.0 for i in range(maxnum_iters)])
    classifier_set = []

    #############################################################################
    #--Hyperparameter tuning: the best number of base learners
    max_map_val = 0.0
    bestnum_baselearner_map = 0
    #############################################################################
    print("Start adaboost learning loops.")
    for i in range(maxnum_iters):
        classifier = RandomForestClassifier(n_estimators=50,
                                            verbose=2,
                                            n_jobs=4,
                                            min_samples_split=10,
                                            random_state=1)
        if i == 0:
            classifier.fit(train_features, train_target)
        else:
            classifier.fit(train_features, train_target, trdata_importance)
        train_predictions = classifier.predict_proba(train_features)[:,1]
        classifier_set.append(classifier)

        ########################################################################
        #--Hyperparameter tuning: the best number of base learners
        # the first method to calculate error rate: absolute value difference
        error_rate = np.dot(trdata_importance,\
                np.abs(train_target-train_predictions))
        # the second method: negative MAP value
        '''author_predictions = defaultdict(list)
        paper_predictions = {}
        for (a_id, p_id), pred, label in zip(train_author_paper_ids,\
                train_predictions, train_target):
            author_predictions[int(a_id)].append([pred, int(p_id), label])

        for author_id in sorted(author_predictions):
            author_predictions[author_id] = reorder(author_predictions[author_id])
            paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
            paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]
        map_val = calcMAP(paper_predictions, train_author_confirmed)
        error_rate = 1 - map_val'''
        # the third method: approximate misclassification error
        '''delta = 0.05
        error_rate = np.dot(trdata_importance,\
                np.abs(train_target-train_predictions)>delta)'''
        ########################################################################

        print "error rate: %f" % error_rate
        model_weights[i] = 1.0/5*np.log((1.0-error_rate)/error_rate)
        print model_weights[i]
        raw_input()
        #model_weights = model_weights / np.sum(model_weights)
        conv_predictions = np.array([(pred-0.5)*2 for pred in train_predictions])
        #for j in range(len(conv_predictions)):
        #    if conv_predictions[j] > 0:
        #        conv_predictions[j] = 1
        #    else:
        #        conv_predictions[j] = -1
        trdata_importance = trdata_importance*np.exp(-model_weights[i]*\
                train_labels*conv_predictions)
        trdata_importance = trdata_importance/np.sum(trdata_importance)

        ########################################################################
        #--Hyperparameter tuning: the best number of base learners
        test_predictions = np.array([0.0 for j in range(len(train_target))])
        for j in range(i+1):
            test_predictions = [pred1+pred2 for pred1,pred2 in\
                    zip(test_predictions, model_weights[j]*\
                    classifier_set[j].predict_proba(test_features)[:,1])]

        author_predictions = defaultdict(list)
        paper_predictions = {}
        for (a_id, p_id), pred in zip(test_author_paper_ids, test_predictions):
            if p_id in test_author_confirmed[int(a_id)]:
                author_predictions[int(a_id)].append([pred, int(p_id), 1])
            else:
                author_predictions[int(a_id)].append([pred, int(p_id), 0])

        for author_id in sorted(author_predictions):
            author_predictions[author_id] = reorder(author_predictions[author_id])
            paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
            paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]

        print("Test the prediction results with MAP metric.")
        map_val = calcMAP(paper_predictions, test_author_confirmed)
        print "Iteration #%d: MAP value --> %f" % (i+1, map_val)
        if map_val > max_map_val:
            max_map_val = map_val
            bestnum_baselearner_map = i+1
        print "Best MAP value --> %f, best number of learners --> %d\n"\
                % (max_map_val, bestnum_baselearner_map)
        #raw_input()
        ########################################################################

    model_weights = model_weights / np.sum(model_weights)
    #============================================================================

    #============================================================================
    #--3.Prediction results on test data
    test_predictions = np.dot(model_weights,\
            np.array([classifier.predict_proba(test_features)[:,1] for\
            classifier in classifier_set]))

    author_predictions = defaultdict(list)
    paper_predictions = {}
    for (a_id, p_id), pred in zip(test_author_paper_ids, test_predictions):
        if p_id in test_author_confirmed[int(a_id)]:
            author_predictions[int(a_id)].append([pred, int(p_id), 1])
        else:
            author_predictions[int(a_id)].append([pred, int(p_id), 0])

    for author_id in sorted(author_predictions):
        author_predictions[author_id] = reorder(author_predictions[author_id])
        paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
        paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]

    print("Test the prediction results with MAP metric.")
    map_val = calcMAP(paper_predictions, test_author_confirmed)
    print("Final MAP value: %f" % map_val)
    #============================================================================

if __name__ == "__main__":
    main()

