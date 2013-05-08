import csv, json
from collections import defaultdict

import data_io

def main():
    print("Loading paths")
    paths = json.loads(open("SETTINGS.json").read())

    print("Getting features for papers gotten from training data partition")
    features_conf = csv.reader(open(paths["trainpos_features"]))
    features_deleted = csv.reader(open(paths["trainneg_features"]))
    #--To be completed--#
    # test

    print("Getting features for valid papers from the database")
    data = csv.reader(open(paths["vali_features"]))
    author_paper_ids = [x[:2] for x in data]
    features = [x[2:] for x in data]

    print("Loading the classifier")
    classifier = data_io.load_model()

    print("Making predictions")
    predictions = classifier.predict_proba(features)[:,1]
    predictions = list(predictions)

    author_predictions = defaultdict(list)
    paper_predictions = {}

    for (a_id, p_id), pred in zip(author_paper_ids, predictions):
        author_predictions[a_id].append((pred, p_id))

    for author_id in sorted(author_predictions):
        paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
        paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]

    print("Writing predictions to file")
    data_io.write_submission(paper_predictions)

if __name__=="__main__":
    main()
