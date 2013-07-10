import pickle
import sys, csv
import numpy as np
from collections import defaultdict

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

def get_train_confirmed(pairs):
    author_confirmed = defaultdict(list)
    for pair in pairs:
        author_confirmed[int(pair[0])].append(int(pair[1]))
    return author_confirmed

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

