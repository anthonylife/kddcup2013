import csv, json
import data_io
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading paths")
    paths = json.loads(open("SETTINGS.json").read())

    print("Getting features for deleted papers from the database")
    features_conf = [feature for feature in csv.reader(open(paths["trainpos_features"]))]
    features_deleted = [feature for feature in csv.reader(open(paths["trainneg_features"]))]

    features = [x[2:] for x in features_deleted + features_conf]
    target = [0 for x in range(len(features_deleted))] + [1 for x in range(len(features_conf))]

    print("Training the Classifier")
    classifier = RandomForestClassifier(n_estimators=50,
                                        verbose=2,
                                        n_jobs=1,
                                        min_samples_split=10,
                                        random_state=1)
    classifier.fit(features, target)

    print("Saving the classifier")
    data_io.save_model(classifier)

if __name__=="__main__":
    main()
