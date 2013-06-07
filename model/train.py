import csv, json, sys
import data_io
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def feature_selection(orig_features, filter_idx_set):
    all_feature_num = len(orig_features[0])
    new_features = np.array([[i for i in range(all_feature_num-len(filter_idx_set))] for j in range(len(orig_features))])
    orig_features = np.array(orig_features)

    idx = 0
    for i in range(all_feature_num):
        if (i+1) in filter_idx_set:
            continue
        new_features[:,idx] = orig_features[:,i]
        idx += 1

    if idx != (all_feature_num - len(filter_idx_set)):
        print "dimension of generated feature matrix is wrong."
        sys.exit(1)

    new_features = new_features.tolist()


def main():
    filter_idx_set = set([11, 14, 64, 16, 67, 13, 60, 50, 10, 63])

    print("Loading paths")
    paths = json.loads(open("SETTINGS.json").read())

    print("Getting features for deleted papers from the disk files")
    features_conf = [feature for feature in csv.reader(open(paths["trainpos_features"]))]
    features_deleted = [feature for feature in csv.reader(open(paths["trainneg_features"]))]
    #features_conf = [feature for feature in csv.reader(open(paths["tc_doc"]))]
    #features_deleted = [feature for feature in csv.reader(open(paths["td_doc"]))]

    features = [x[2:] for x in features_deleted + features_conf]
    features = feature_selection(features, filter_idx_set)
    target = [0 for x in range(len(features_deleted))] + [1 for x in range(len(features_conf))]

    print("Training the Classifier")
    classifier = RandomForestClassifier(n_estimators=360,
                                        verbose=2,
                                        n_jobs=4,
                                        min_samples_split=10,
                                        random_state=1)

    classifier.fit(features, target)

    print("Saving the classifier")
    data_io.save_model(classifier)

if __name__=="__main__":
    main()
