#!/usr/bin/env python
#encoding=utf8

# Copyright: for KDDCup 2013 track 1, team group work.
# Author: anthonylife
#
# General Ideas:
#   Eliminate one feature each step recursively; Choose feature subset which gets
#   the best result on cross-validation test set.
#
# Selection Result Output:
#   feature_support: "True" for feature saved; "False" for feature removed.
#   feature_importance: Orders of features accroding to their importance.

import csv, json, sys, random
sys.path.append("../")
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from evaluation.metrics import eval_map


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


def feature_strip(orig_features, filter_idx_set):
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


def update_idx_map(feature_idx_map, removed_feature_idx, cur_feature_num):
    if removed_feature_idx > cur_feature_num-1:
        print 'feature number is invalid.'
        sys.exit(1)

    for i in range(removed_feature_idx, cur_feature_num-1):
        feature_idx_map[i] = feature_idx_map[i+1]

    return feature_idx_map


def main():
    cv_num = 5
    terminal_dim_num = 0
    best_map = 0.0
    best_subset_num = 0
    strip_feature_idx = set([])

    print("Loading paths")
    paths = json.loads(open("SETTINGS.json").read())

    #===================================================================================
    #-1.Preparation for feature selection.
    print("Getting features for deleted papers from the disk files")
    #features_conf = [feature for feature in csv.reader(open(paths["trainpos_features"]))]
    #features_deleted = [feature for feature in csv.reader(open(paths["trainneg_features"]))]
    features_conf = [feature for feature in csv.reader(open(paths["tc_all_features"]))]
    features_deleted = [feature for feature in csv.reader(open(paths["td_all_features"]))]

    author_features, author_set = ct_author_feature_dict(features_conf, features_deleted)
    cv_author = partition_author(author_set, cv_num)
    #===================================================================================

    #===================================================================================
    #-2.Do feature selection iteratively.
    all_dim_num = len(features_conf[0][2:])

    # selected feature subset with best evaluation result on cross-validation test
    # feature_support = [True for i in range(all_dim_num)]
    # evaluation result for each subset features
    feature_eval_result = [-1.0 for i in range(all_dim_num)]
    # rand order for each feature dimension accroding to its importance
    feature_importance_rank = [0 for i in range(all_dim_num)]
    # mapping relation of original feature number to new number of current feature subset
    feature_idx_map = [i for i in range(all_dim_num)]

    # model
    classifier = RandomForestClassifier(n_estimators=20,
                                        verbose=2,
                                        n_jobs=4,
                                        min_samples_split=10,
                                        random_state=1,
                                        compute_importances=True)

    cv_features_set = ct_feature_set(author_features, cv_author)
    for i in range(all_dim_num, terminal_dim_num, -1):
        print 'Loop each feature with %d features left.' % i
        feature_idx = [temp_idx for temp_idx in range(i)]
        feature_importance = [[idx, 0.0] for idx in feature_idx]
        map_val = 0.0
        dim = 0
        for j in range(cv_num):
            print '\nCross validation iteration #%d.\n' % (j+1)
            train_features = []
            train_target = []
            test_features = []
            test_target = []

            # construct training and test features and their corresponding targets
            for k in range(cv_num):
                if k != j:
                    train_features += cv_features_set[j][0]
                    train_target += cv_features_set[j][1]
                else:
                    test_features += cv_features_set[j][0]
                    test_target += cv_features_set[j][1]

            # model training
            filter_train_features = feature_strip([feature[2:] for feature in train_features], strip_feature_idx)
            dim = len(filter_train_features[0])
            classifier.fit(filter_train_features, train_target)
            feature_importance = [[entry[0], entry[1]+score] for entry, score \
                    in zip(feature_importance, classifier.feature_importances_)]

            # model prediction
            filter_test_features = feature_strip([feature[2:] for feature in test_features], strip_feature_idx)
            predictions = classifier.predict_proba(filter_test_features)[:,1]

            author_paper_ids = [feature[:2] for feature in test_features]
            predictions = list(predictions)
            author_predictions = defaultdict(list)
            for (a_id, p_id), pred, target in zip(author_paper_ids, predictions, test_target):
                author_predictions[a_id].append([pred, p_id, target])

            # valuation
            map_val += eval_map(author_predictions)

        feature_eval_result[i-1] = map_val*1.0/cv_num
        print feature_eval_result[i-1]
        print feature_importance
        if feature_eval_result[i-1] > best_map:
            best_map = feature_eval_result[i-1]
            best_subset_num = i
        feature_importance = sorted(feature_importance, key=lambda x:x[1], reverse=False)
        cur_idx = feature_idx_map[feature_importance[0][0]]
        feature_importance_rank[cur_idx] = i
        strip_feature_idx.add(cur_idx)
        print dim
        print strip_feature_idx
        feature_idx_map = update_idx_map(feature_idx_map, feature_importance[0][0], i)
        print feature_importance
        print feature_idx_map
        print "\nBest result"
        print best_map
        print best_subset_num
        #raw_input()
    #===================================================================================

    #===================================================================================
    #-3.Result output.
    writer = open(paths["my_selection_result"], "w")
    writer.write("Evaluation result for each feature subset.\n")
    for i in range(all_dim_num):
        if feature_eval_result[i] >= 0:
            writer.write("%d:%.8f " % (i+1, feature_eval_result[i]))
    writer.write("\n\n")
    writer.write("Rank order for each feature according to its importance.\n")
    for i in range(all_dim_num):
        writer.write("%d:%d " % (i+1, feature_importance_rank[i]))
    writer.write("\n")
    writer.close()
    #===================================================================================

if __name__ == "__main__":
    main()
