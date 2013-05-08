#!/usr/bin/env python
#encoding=utf8

import csv
import sys
import json
import random
from mkTitleSim import TitleSim


def list_to_dict(ins):
    item_dict = {}
    for entry in ins:
        if entry[0] in item_dict:
            item_dict[entry[0]].append(entry[1:])
        else:
            item_dict[entry[0]] = [entry[1:]]
    return item_dict


def dict_to_list(item_dict, item_id):
    item_list = []
    for key in item_id:
        if key not in item_dict:
            print "Key error."
            sys.exit(1)
        for entry in item_dict[key]:
            item_list.append([key] + entry)
    return item_list


def sub_dict(somedict, somekeys):
    return dict([(key, somedict.get(key)) for key in somekeys])


def select_doc(prob_dict1, prob_dict2=None):
    author_doc = {}
    for author_key in prob_dict1.keys():
        if prob_dict2 != None:
            if author_key not in prob_dict2.keys():
                print 'Key not match.'
                sys.exit(1)
            doc_list = prob_dict1[author_key] + prob_dict2[author_key]
            doc_list = sorted(doc_list, key = lambda x:x[1], reverse=True)
            topk = len(doc_list)/3+1
            author_doc[author_key] = [entry[0] for entry in doc_list[:topk]]
        else:
            doc_list = prob_dict1[author_key]
            doc_list = sorted(doc_list, key = lambda x:x[1], reverse=True)
            topk = len(doc_list)/3+1
            author_doc[author_key] = [entry[0] for entry in doc_list[:topk]]
    return author_doc


def mkpairfeatures(titleSim, paths, pos_dict, neg_dict, author_set,\
        pos_doc, neg_doc):
    postr_subdict = sub_dict(pos_dict, author_set)
    negtr_subdict = sub_dict(neg_dict, author_set)
    postr_prob = titleSim.label_predict(postr_subdict)
    negtr_prob = titleSim.label_predict(negtr_subdict)
    author_doc = select_doc(postr_prob, negtr_prob)
    features_conf = dict_to_list(pos_dict, author_set)
    features_deleted = dict_to_list(neg_dict, author_set)
    pos_titlesim = titleSim.calsim(author_doc, [train_pair[0:2] for train_pair in features_conf])
    neg_titlesim = titleSim.calsim(author_doc, [train_pair[0:2] for train_pair in features_deleted])

    newfeatures_conf = [entry1+entry2 for entry1,entry2 in zip(features_conf, pos_titlesim)]
    newfeatures_deleted = [entry1+entry2 for entry1,entry2 in zip(features_deleted, neg_titlesim)]
    writer = csv.writer(open(paths[pos_doc], 'w'), lineterminator="\n")
    writer.writerows(newfeatures_conf)
    writer = csv.writer(open(paths[neg_doc], 'w'), lineterminator="\n")
    writer.writerows(newfeatures_deleted)


def mksinglefeatures(titleSim, paths, test_features, test_dict, test_doc):
    test_prob = titleSim.label_predict(test_dict)
    author_doc = select_doc(test_prob)
    test_titlesim = titleSim.calsim(author_doc, [test_pair[0:2] for test_pair in test_features])
    test_newfeatures = [entry1+entry2 for entry1,entry2 in zip(test_features, test_titlesim)]
    writer = csv.writer(open(paths[test_doc], 'w'), lineterminator="\n")
    writer.writerows(test_newfeatures)


def main():
    # reading paths
    print "Reading file path from the configuration file 'SETTINGS.json'"
    paths = json.loads(open("SETTINGS.json").read())
    postr_doc = paths["postr_doc"]
    negtr_doc = paths["negtr_doc"]
    vali_doc  = paths["vali_doc"]

    # loading instances
    print "Loading labeled instances."
    postr_ins = [ins[:-1] for ins in csv.reader(open(postr_doc))]
    negtr_ins = [ins[:-1] for ins in csv.reader(open(negtr_doc))]
    vali_ins = [ins[:-1] for ins in csv.reader(open(vali_doc))]
    #--test--#
    #print postr_ins[:3]
    #raw_input()

    # convert list to dictionary which regard "author id" as key
    postr_dict = list_to_dict(postr_ins)
    negtr_dict = list_to_dict(negtr_ins)
    vali_dict  = list_to_dict(vali_ins)
    #--test--#
    #print postr_dict["826"]
    #raw_input()

    # randomly partition users
    authors = list(set([entry[0] for entry in postr_ins]))
    random.shuffle(authors)
    init_set = authors[0:len(authors)/3]
    train_set = authors[len(authors)/3+1:len(authors)*3/4]
    test_set = authors[len(authors)*3/4+1:len(authors)]

    # making the first feature: title similarity
    print "Making title similarity features."
    features_conf = dict_to_list(postr_dict, init_set)
    features_deleted = dict_to_list(negtr_dict, init_set)
    #--test--#
    #print features_conf[:3]
    #raw_input()
    titleSim = TitleSim([feature[2:] for feature in features_conf],\
            [feature[2:] for feature in features_deleted])

    ## make features for training pairs in positive ins and negative ins
    mkpairfeatures(titleSim,paths,postr_dict,negtr_dict,train_set,\
            "new_postr_doc","new_negtr_doc")

    ## make features for test pairs in positive ins and negative ins
    mkpairfeatures(titleSim,paths,postr_dict,negtr_dict,test_set,\
            "new_poste_doc","new_negte_doc")

    ## make features for validation pairs
    mksinglefeatures(titleSim, paths, vali_ins, vali_dict, "new_vali_doc")



if __name__ == "__main__":
    main()
