import csv
import json
import pickle

def paper_ids_to_string(ids):
    return " ".join([str(x) for x in ids])

def save_model(model):
    paths = json.loads(open("SETTINGS.json").read())
    out_path = paths["model_path"]
    pickle.dump(model, open(out_path, "w"))

def load_model():
    paths = json.loads(open("SETTINGS.json").read())
    in_path = paths["model_path"]
    return pickle.load(open(in_path))

def write_submission(predictions):
    paths = json.loads(open("SETTINGS.json").read())
    submission_path = paths["submission_path"]
    rows = [(prediction[0], paper_ids_to_string(prediction[1:])) for prediction in predictions]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    writer.writerow(("AuthorId", "PaperIds"))
    writer.writerows(rows)
