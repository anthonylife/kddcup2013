#!/usr/bin/env python
#encoding=utf8

import sys
sys.path.append('../')
import csv, json
from nltk.stem import PorterStemmer
from skleanr.ensemble import RandomForestClassifier

class TitleSim:
    def __init__(self, features_conf, features_deleted):
        print 'Start initialization'

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

        # loading setting file
        self.paths = json.loads(open("SETTINGS.json").read())

        # loading word map of titles
        self.wordmap = self.load_titlemap()

        # do other initializations
        self.stemmer = PorterStemmer()
        print 'End initialization'


    def label_predict(self, fea_dict):
        # fea_dict is a dictionary whose key is 'user id'
        prob_dict = {}
        for key in fea_dict:
            features = [feature[1:] for feature in fea_dict[key]]
            predictions = self.classifier.predict_proba(features)[:,1]
            prob_dict[key]=[(item[0],prob) for item,prob in zip(fea_dict[key],predictions)]
        return prob_dict


    def load_titlemap(self):
        return dict([(entry[0],entry[1]) for entry in \
                csv.reader(open(self.paths["title_wordmap"]))])

    def calsim(self, author_doc, pairs):
        # calculate the similarity between titles
        title_features = []
        for pair in pairs:
            if pair[0] not in author_doc:
                print 'Key error.'
                sys.exit(1)
            title_features += self.calpairsim(author_doc[pair[0]], pair[1])

        return title_features

    def calpairsim(self, doclist, target_doc):
        author_words = {}
        for doc in doclist:
            words = self.paper[doc].lower().split(' ')
            for word in words:
                stemmed_word = self.stemmer.stem_word(word)
                if stemmed_word in self.wordmap:
                    if stemmed_word in author_words:
                        author_words[stemmed_word] += 1
                    else:
                        author_words[stemmed_word] = 1

        doc_words = {}
        words = self.paper[target_doc].lower().split(' ')
        for word in words:
            stemmed_word = self.stemmer.stem_word(word)
            if stemmed_word in self.wordmap:
                if stemmed_word in doc_words:
                    doc_words[stemmed_word] += 1
                else:
                    doc_words[stemmed_word] = 1

        # number of common words
        comm_num = len(set(author_words.keys()) & set(doc_words.keys()))

        # pearson coefficient
        pearson = comm_num*1.0/ (len(set(author_words.keys())) + len(set(doc_words.keys())))

        return [comm_num, pearson]
