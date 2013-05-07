#!/usr/bin/env python
#encoding=utf8

import csv, json
from skleanr.ensemble import RandomForestClassifier

class TitleSim:
    def __init__(self, features_conf, features_deleted):
        # initial model training
        features = features_deleted + features_conf
        target = [0 for x in range(len(features_deleted))] +\
                [1 for x in range(len(features_conf))]
        self.classifier = RandomForestClassifier(n_estimators=50,
                                            verbose=2,
                                            n_jobs=1,
                                            min_samples_split1=10,
                                            random_state=1)
        self.classifier.fit(features, target)

        # loading relational data which will be used
        paths = json.loads(open("SETTINGS.json").read())
        paper_doc = paths["paper_doc"]
        self.paper = dict([(entry[0], entry[1]) for entry in csv.reader(open(paper_doc))])


    def label_predict(self, fea_dict):
        # fea_dict is a dictionary whose key is 'user id'
        prob_dict = {}

        for key in fea_dict:
            features = [feature[1:] for feature in fea_dict[key]]

    def calsim(self, )
