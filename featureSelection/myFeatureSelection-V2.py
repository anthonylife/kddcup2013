#!/usr/bin/env python
#encoding=utf8

# Copyright: for KDDCup 2013 track 1, team group work.
# Author: anthonylife
#
# Function: checking the performance of our classifier on training and
#           validation data.
#

import csv, json, sys, random
sys.path.append("../")
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from evaluation.MAPUtil import calcMAP
from ensembleRF.util import feature_selection, get_confirmed_paper
from ensembleRF.util import get_train_confirmed, reorder


def ct_author_feature_dict(features_conf, features_deleted, features_valid,\
        valid_author_confirmed):
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

    for entry in features_valid:
        if entry[0] not in author_features:
            author_features[entry[0]]['1'] = []
            author_features[entry[0]]['0'] = []
        if entry[1] in valid_author_confirmed[entry[0]]:
            author_features[entry[0]]['1'].append(entry[1:])
        else:
            author_features[entry[0]]['0'].append(entry[1:])
        author_set.add(entry[0])

    return author_features, author_set


def partition_author(author_set, cv_num):
    cv_author = []
    author_set = list(author_set)
    random.shuffle(author_set)

    author_num  = len(author_set)
    sep_num_set = [0]
    for i in range(cv_num):
        if i+1 == cv_num:
            sep_num_set.append(author_num)
        else:
            sep_num_set.append(author_num*(i+1)/cv_num)
        cv_author.append(author_set[sep_num_set[i]:sep_num_set[i+1]])

    return cv_author


def ct_feature_set(author_features, cv_author, author_confirmed):
    ''' Return features with format: "authorid, paperid, features1, feature2, ..."
    '''
    cv_features = []
    cv_author_confirmed = []

    for sub_author_set in cv_author:
        features = []
        target = []
        sub_author_confirmed = defaultdict(list)
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
            sub_author_confirmed[author] = author_confirmed[author]
        cv_features.append([features, target])
        cv_author_confirmed.append(sub_author_confirmed)

    return cv_features, cv_author_confirmed


def update_idx_map(feature_idx_map, removed_feature_idx, cur_feature_num):
    if removed_feature_idx > cur_feature_num-1:
        print 'feature number is invalid.'
        sys.exit(1)

    for i in range(removed_feature_idx, cur_feature_num-1):
        feature_idx_map[i] = feature_idx_map[i+1]

    return feature_idx_map


def load_cv_author(doc_path):
    cv_author = []
    for authorids in csv.reader(open(doc_path)):
        cv_author.append(map(int, authorids))
    return cv_author

def main():
    cv_num = 3
    terminal_dim_num = 0
    best_map = 0.0
    best_subset_num = 0
    filter_feature_idx = set([])

    print("Loading paths")
    paths = json.loads(open("SETTINGS.json").read())

#================================================================================
    #-------1.Preparation for feature selection.
    print("Getting features for deleted papers from the disk files")
    features_conf = [map(int, feature[:2])+map(float, feature[2:]) for feature in\
            csv.reader(open(paths["tc_all_features"]))]
    features_deleted = [map(int,feature[:2])+map(float,feature[2:]) for feature in\
            csv.reader(open(paths["td_all_features"]))]
    features_valid = [map(int, feature[:2])+map(float, feature[2:]) for feature in\
            csv.reader(open(paths["v_all_features"]))]
    train_author_confirmed = get_train_confirmed([map(int, pair[:2]) for pair in\
            features_conf])
    valid_author_confirmed = get_confirmed_paper(paths["vali_solution"])
    author_confirmed = dict(train_author_confirmed.items()+\
            valid_author_confirmed.items())

    print("Randomly partition authors.")
    author_features, author_set = ct_author_feature_dict(features_conf,\
            features_deleted, features_valid, valid_author_confirmed)
    #cv_author = partition_author(author_set, cv_num)
    cv_author = load_cv_author(paths["cv_author"])
    cv_features_set, cv_author_confirmed = ct_feature_set(author_features,\
            cv_author, author_confirmed)
#================================================================================

#================================================================================
    #-------2.Do feature selection iteratively.
    all_dim_num = len(features_conf[0][2:])
    # evaluation result for each subset features
    feature_eval_result = [-1.0 for i in range(all_dim_num)]
    # rand order for each feature dimension accroding to its importance
    feature_importance_rank = [0 for i in range(all_dim_num)]
    # mapping relation of original feature ID to ID of current feature subset
    feature_idx_map = [i for i in range(all_dim_num)]

    classifier = RandomForestClassifier(n_estimators=360,
                                        verbose=2,
                                        n_jobs=4,
                                        min_samples_split=10,
                                        random_state=1,
                                        compute_importances=True)

    for ii in range(all_dim_num, terminal_dim_num, -1):
        print 'Loop each feature with %d features left.' % ii
        feature_idx = [temp_idx for temp_idx in range(ii)]
        feature_importance = [[idx, 0.0] for idx in feature_idx]
        dim = 0

        cv_map_val = []
        for i in range(cv_num):
            print '\nCross validation iteration #%d.\n' % (i+1)
            train_features = []
            train_target = []
            test_features = []
            test_target = []
            for k in range(cv_num):
                if k != i:
                    train_features += cv_features_set[k][0]
                    train_target += cv_features_set[k][1]
                else:
                    test_features += cv_features_set[k][0]
                    test_target += cv_features_set[k][1]

            print("Model training")
            filter_train_features = feature_selection([feature[2:] for feature in\
                    train_features], filter_feature_idx)
            dim = len(filter_train_features[0])
            classifier.fit(filter_train_features, train_target)
            feature_importance = [[entry[0], entry[1]+score] for entry, score \
                    in zip(feature_importance, classifier.feature_importances_)]

            print("Model prediction")
            filter_test_features = feature_selection([feature[2:] for feature in\
                    test_features], filter_feature_idx)
            test_predictions = classifier.predict_proba(filter_test_features)[:,1]
            test_author_paper_ids = [map(int, feature[:2]) for feature in\
                    test_features]

            author_predictions = defaultdict(list)
            paper_predictions = {}
            for (a_id, p_id), pred in zip(test_author_paper_ids, test_predictions):
                if p_id in cv_author_confirmed[i][int(a_id)]:
                    author_predictions[int(a_id)].append([pred, int(p_id), 1])
                else:
                    author_predictions[int(a_id)].append([pred, int(p_id), 0])

            for author_id in sorted(author_predictions):
                author_predictions[author_id] =\
                        reorder(author_predictions[author_id])
                paper_ids_sorted = sorted(author_predictions[author_id],\
                        reverse=True)
                paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]

            print("Test the prediction results with MAP metric.")
            map_val = calcMAP(paper_predictions, cv_author_confirmed[i])
            #map_val = eval_map(author_predictions)
            print '\nCross validation iteration #%d.\n' % (i+1)
            print("MAP value: %f" % map_val)
            cv_map_val.append(map_val)

        map_val = 0.0
        for j in range(cv_num):
            map_val += cv_map_val[j]
            print("CV number #%d, MAP value: %f\n" % (j+1, cv_map_val[j]))
        feature_eval_result[ii-1] = map_val/cv_num
        print dim
        if feature_eval_result[ii-1] > best_map:
            best_map = feature_eval_result[ii-1]
            best_subset_num = ii
        feature_importance = sorted(feature_importance, key=lambda x:x[1], reverse=False)
        cur_idx = feature_idx_map[feature_importance[0][0]]
        feature_importance_rank[cur_idx] = ii
        filter_feature_idx.add(cur_idx)
        feature_idx_map=update_idx_map(feature_idx_map,feature_importance[0][0],ii)
        print feature_importance
        print feature_idx_map
        print "\nBest result"
        print best_map
        print best_subset_num
#================================================================================

#================================================================================
    #-------3.Result output.
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
#================================================================================

if __name__ == "__main__":
    main()

