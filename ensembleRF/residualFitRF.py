#!/usr/bin/env python
#encoding=utf8

# Copyright: for KDDCup 2013 track 1, team group work.
# Author: anthonylife
#
# Idea: Combine random forests sequentially by fitting the current residual.
#       This is the method of  modify the objective funtions to improve the fusion result.
# Procedures: 1.preparation data;
#             2.loops:
#               2.1 iteratively training random forest on current resudals;
#               2.2 save models;
#               2.2 evaluation on validation data;
#3.model combination by linear regression.

import csv, json, sys
sys.path.append("../")
import pickle
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from evaluation.MAPUtil import calcMAP
from sklearn.metrics import mean_squared_error

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

def get_confirmed_paper(filepath):
   key_rank = {}
   for row in csv.reader(open(filepath)):
      key = int(row[0])
      ids = map(int, row[1].split())
      key_rank[key] = ids
   return key_rank

def reorder(prob_id):
    ids = set()
    repeated_ids = set()
    for entry in prob_id:
        if entry[1] in ids:
            repeated_ids.add(entry[1])
        else:
            ids.add(entry[1])

    tag = {}
    for entry in repeated_ids:
        tag[entry] = 1

    for i, entry in enumerate(prob_id):
        if entry[1] in repeated_ids:
            if tag[entry[1]] == 1:
                tag[entry[1]] = 0
            elif tag[entry[1]] == 0:
                prob_id[i][0] = -1

    return prob_id

def save_model(model, model_path):
    pickle.dump(model, open(model_path, "w"))

def load_model(model_path):
    return pickle.load(open(model_path))


def main():
    #============================================================================
    #--0.Preparation for training model.
    maxnum_baselearner = 5
    filter_idx_set = set([11, 14, 64, 16])
    paths = json.loads(open("SETTINGS.json").read())

    print("Getting features for deleted papers from the disk files")
    features_conf = [feature for feature in \
            csv.reader(open(paths["trainpos_features"]))]
    features_deleted = [feature for feature in \
            csv.reader(open(paths["trainneg_features"]))]
    train_features = [map(float, x[2:]) for x in features_deleted + features_conf]
    train_features = feature_selection(train_features, filter_idx_set)
    train_target = [0 for x in range(len(features_deleted))] + [1 for x in range(len(features_conf))]

    features_valid = [feature for feature in \
            csv.reader(open(paths["vali_features"]))]
    test_features = [map(float, x[2:]) for x in features_valid]
    test_features = feature_selection(test_features, filter_idx_set)
    author_confirmed = get_confirmed_paper(paths["vali_solution"])
    test_author_paper_ids = [x[:2] for x in features_valid]
    #============================================================================

    #============================================================================
    #--2.Model training (Random forest)
    base_learner_set = []
    max_map_val = 0.0
    bestnum_baselearner_map = 0
    min_mse_val = 1e5
    bestnum_baselearner_mse = 0
    dynamic_target = [i for i in train_target]
    for i in range(maxnum_baselearner):
        train_predictions = [0.0 for j in range(len(train_target))]
        test_predictions = [0.0 for j in range(len(test_features))]

        classifier = RandomForestRegressor(n_estimators=50,
                                           verbose=2,
                                           n_jobs=4,
                                           min_samples_split=10,
                                           random_state=1)
        classifier.fit(train_features, dynamic_target)
        base_learner_set.append(classifier)

        for base_learner in base_learner_set:
            unit_predictions = base_learner.predict(train_features)
            train_predictions = [pred1+pred2 for pred1, pred2 in\
                    zip(train_predictions, list(unit_predictions))]
            unit_predictions = base_learner.predict(test_features)
            test_predictions = [pred1+pred2 for pred1, pred2 in\
                    zip(test_predictions, list(unit_predictions))]

        dynamic_target = [target-pred for target, pred in\
                zip(train_target, train_predictions)]

        author_predictions = defaultdict(list)
        paper_predictions = {}
        mse_predictions = []
        mse_labels = []
        for (a_id, p_id), pred in zip(test_author_paper_ids, test_predictions):
            if p_id in author_confirmed[int(a_id)]:
                author_predictions[int(a_id)].append([pred, int(p_id), 1])
                mse_labels.append(1)
            else:
                author_predictions[int(a_id)].append([pred, int(p_id), 0])
                mse_labels.append(0)
            mse_predictions.append(pred)

        for author_id in sorted(author_predictions):
            author_predictions[author_id] = reorder(author_predictions[author_id])
            paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
            paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]

        print("Test the prediction results with MAP metric.")
        map_val = calcMAP(paper_predictions, author_confirmed)
        mse_val = mean_squared_error(mse_labels, mse_predictions)
        print "Iteration #%d: MAP value --> %f, MSE value -->%f\n"\
                % (i+1, map_val, mse_val)
        if map_val > max_map_val:
            max_map_val = map_val
            bestnum_baselearner_map = i+1
        print "Best MAP value --> %f, best number of learners --> %d\n"\
                % (max_map_val, bestnum_baselearner_map)
        if mse_val < min_mse_val:
            min_mse_val = mse_val
            bestnum_baselearner_mse = i+1
        print "Best MSE value --> %f, best number of learners --> %d\n"\
                % (min_mse_val, bestnum_baselearner_mse)


if __name__ == "__main__":
    main()


