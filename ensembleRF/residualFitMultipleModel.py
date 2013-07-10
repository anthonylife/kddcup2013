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
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, LinearRegression
from evaluation.MAPUtil import calcMAP
from sklearn.metrics import mean_squared_error
from util import feature_selection,get_confirmed_paper,reorder


def main():
    #============================================================================
    #--0.Preparation for training model.
    maxnum_baselearner = 2
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
    classifier = RandomForestClassifier(n_estimators=360,
                                        verbose=2,
                                        n_jobs=4,
                                        min_samples_split=10,
                                        random_state=1)
    base_learner_set.append(classifier)

    regressor = GradientBoostingRegressor(loss='ls',
                                        learning_rate=0.1,
                                        n_estimators=450,
                                        max_depth=2)
    base_learner_set.append(regressor)

    max_map_val = 0.0
    bestnum_baselearner_map = 0
    min_mse_val = 1e5
    bestnum_baselearner_mse = 0
    dynamic_target = [i for i in train_target]
    for i in range(maxnum_baselearner):
        train_predictions = [0.0 for j in range(len(train_target))]
        test_predictions = [0.0 for j in range(len(test_features))]

        base_learner_set[i].fit(train_features, dynamic_target)

        for j in range(i+1):
            if j == 0:
                unit_predictions = base_learner_set[j].predict_proba(train_features)[:,1]
            elif j >= 1:
                unit_predictions = base_learner_set[j].predict(train_features)
            print unit_predictions[0:5]
            print unit_predictions[100:105]
            train_predictions = [pred1+pred2 for pred1, pred2 in\
                    zip(train_predictions, list(unit_predictions))]
            print train_predictions[0:5]
            print train_predictions[100:105]

            if j == 0:
                unit_predictions = base_learner_set[j].predict_proba(test_features)[:,1]
            elif j >= 1:
                unit_predictions = base_learner_set[j].predict(test_features)
            test_predictions = [pred1+pred2 for pred1, pred2 in\
                    zip(test_predictions, list(unit_predictions))]

        dynamic_target = [target-pred for target, pred in\
                zip(train_target, train_predictions)]
        print dynamic_target[:10]

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

    pickle.dump(base_learner_set[0], open('./rf-model.pickle', "w"))
    pickle.dump(base_learner_set[1], open('./gbrt-model.pickle', "w"))


if __name__ == "__main__":
    main()


