import csv, json
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading paths")
    paths = json.loads(open("SETTINGS.json").read())

    print("Getting features for deleted papers from the disk files")
    features_conf = [feature for feature in csv.reader(open(paths["trainpos_features"]))]
    features_deleted = [feature for feature in csv.reader(open(paths["trainneg_features"]))]
    features = [x[2:] for x in features_deleted + features_conf]
    target = [0 for x in range(len(features_deleted))] + [1 for x in range(len(features_conf))]

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

