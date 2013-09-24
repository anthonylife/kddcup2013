import csv, json
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading paths")
    paths = json.loads(open("SETTINGS.json").read())

    print("Getting features for deleted papers from the disk files")
    tc_features = [feature for feature in\
            csv.reader(open(paths["trainpos_features"]))]
    td_features = [feature for feature in\
            csv.reader(open(paths["trainneg_features"]))]
    vc_features = [feature for feature in\
            csv.reader(open(paths["validpos_features"]))]
    vd_features = [feature for feature in\
            csv.reader(open(paths["validneg_features"]))]
    features = [x[2:] for x in td_features+tc_features+vd_features+vc_features]
    target = [0 for x in range(len(td_features))] +\
            [1 for x in range(len(tc_features))] +\
            [0 for x in range(len(vd_features))] +\
            [1 for x in range(len(vc_features))]

    print("Training the Classifier")
    classifier = RandomForestClassifier(n_estimators=360,
                                        verbose=2,
                                        n_jobs=4,
                                        min_samples_split=10,
                                        random_state=1,
                                        compute_importances=True)

    classifier.fit(features, target)

    print("Output feature importance.")
    feature_idx = [i for i in range(len(features[0]))]
    feature_importance = [[idx, score] for idx, score in zip(feature_idx, classifier.feature_importances_)]

    feature_importance = sorted(feature_importance, key=lambda x:x[1], reverse=True)
    print feature_importance

if __name__=="__main__":
    main()
