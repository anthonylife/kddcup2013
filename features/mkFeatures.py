#!/usr/bin/env python
#encoding=utf8

import csv
import sys
import json
import random


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


def main():
    # reading paths
    print "Reading file path from the configuration file 'SETTINGS.json'"
    paths = json.loads(open("SETTINGS.json").read())
    postr_doc = paths["postr_doc"]
    negtr_doc = paths["negtr_doc"]
    vali_doc  = paths["vali_doc"]

    print "Loading labeled instances."
    postr_ins = [ins[:-1] for ins in csv.reader(open(postr_doc))]
    negtr_ins = [ins[:-1] for ins in csv.reader(open(negtr_doc))]
    vali_ins = [ins[:-1] for ins in csv.reader(open(vali_doc))]

    # convert list to dictionary which regard "author id" as key
    postr_dict = list_to_dict(postr_ins)
    negtr_dict = list_to_dict(negtr_ins)
    vali_dict  = list_to_dict(vali_ins)

    # randomly partition users
    authors = set([entry[0] for entry in postr_ins])
    random.shuffle(authors)
    init_set = authors[0:len(authors)/3]
    train_set = authors[len(authors)/3+1:len(authors)*2/3]
    test_set = authors[len(authors)*2/3+1:len(authors)]

    print "Making title similarity features."
    features_conf = dict_to_list(postr_dict, init_set)
    features_deleted = dict_to_list(negtr_dict, init_set)


if __name__ == "__main__":
    main()
