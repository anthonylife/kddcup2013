#!/usr/bin/env python
#encoding=utf8

import csv
import sys
import json
import random
from collections import defaultdict
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
            topk = len(doc_list)/5+1
            author_doc[author_key] = [entry[0] for entry in doc_list[:topk]]
        else:
            doc_list = prob_dict1[author_key]
            doc_list = sorted(doc_list, key = lambda x:x[1], reverse=True)
            topk = len(doc_list)/5+1
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


def load_year_info(author_table1, author_table2, author_table3):
    '''load two tabels, return three dictionary
    '''
    author_paper_year = defaultdict(dict)
    author_year_cnt = defaultdict(dict)
    paper_author = defaultdict(set)
    filter_author_paper = defaultdict(set)

    author_list1 = [entry for entry in csv.reader(open(author_table1))]
    author_list2 = [entry for entry in csv.reader(open(author_table2))]
    author_list3 = [entry for entry in csv.reader(open(author_table3))]

    for entry in author_list1[1:]:
        # author as key
        if entry[2] and int(entry[2]) != 0:
            author_paper_year[entry[0]][entry[1]] = int(entry[2])
        else:
            author_paper_year[entry[0]][entry[1]] = 0
        # paper as key
        paper_author[entry[1]].add(entry[0])

    for entry in author_list2[1:]:
        if entry[1] and int(entry[1]) != 0:
            author_year_cnt[entry[0]][int(entry[1])] = int(entry[2])
        else:
            author_year_cnt[entry[0]][entry[1]] = 0

    for entry in author_list3[1:]:
        filter_author_paper[entry[0]].add(entry[1])

    return (author_paper_year, author_year_cnt, paper_author, filter_author_paper)


def make_year_feature(author_paper_year, author_year_cnt, paper_author, \
        filter_author_paper, pairs):
    time_fea_num = 19
    features = [[-1 for i in range(time_fea_num)] for i in range(len(pairs))]

    for i, pair in enumerate(pairs):
        yearlist = sorted(author_year_cnt[pair[0]].items(), key = lambda x:x[0])
        if len(yearlist) == 0:
            continue

        # 1. time gap between the starting publition time of author and
        #    time of this paper;

        first_year = find_first_year(yearlist)
        if first_year == -1:
            features[i][0] = -1
        else:
            features[i][0] = abs(first_year - author_paper_year[pair[0]][pair[1]])
            if yearlist[0] == 0 or author_paper_year[0] == 0:
                features[i][0] = -1
            elif features[i][0] > 500:
                features[i][0] = -1
        # 2. number of papers the author published in the same year as
        #    target paper;
        year = author_paper_year[pair[0]][pair[1]]
        if year == 0:
            features[i][1] = -1
        else:
            features[i][1] = author_year_cnt[pair[0]][year]
        # 3. precentage of papers publised in the same year as target paper;
        year_count = [entry[1] for entry in author_year_cnt[pair[0]].items()]
        sum_count = sum(year_count)
        if year == 0:
            features[i][2] = 0
        else:
            features[i][2] = author_year_cnt[pair[0]][year]*1.0/sum_count
        # 4. time gap between the year of most publised paper and the year of
        #    target paper;
        (max_year, min_year) = find_minmax_year_cnt(author_year_cnt[pair[0]].items())
        if max_year == -1:
            features[i][3] = -1
        else:
            features[i][3] = abs(max_year - year)
            if yearlist[0] == 0 or author_paper_year[0] == 0:
                features[i][3] = -1
            elif features[i][3] > 500:
                features[i][3] = -1
        # 5. time gap between the year of least publised paper and the year of
        #    target paper;
        if min_year == -1:
            features[i][4] = -1
        else:
            features[i][4] = abs(min_year - year)
            if yearlist[0] == 0 or author_paper_year[0] == 0:
                features[i][4] = -1
            elif features[i][4] > 500:
                features[i][4] = -1

        # 6. number of co-author in current year
        count = make_year_coauthor_feature(0, author_paper_year, paper_author, pair,\
                filter_author_paper)
        features[i][5] = count
        if len(paper_author[pair[1]]) > 1:
            features[i][6] = count*1.0/len(paper_author[pair[1]])
        else:
            features[i][6] = -1
        # 7. number of co-author one year before or after current year
        count = make_year_coauthor_feature(1, author_paper_year, paper_author, pair,\
                filter_author_paper)
        features[i][7] = count
        if len(paper_author[pair[1]]) > 1:
            features[i][8] = count*1.0/len(paper_author[pair[1]])
        else:
            features[i][8] = -1
        # 8. number of co-author two year before current year
        count = make_year_coauthor_feature(-2, author_paper_year, paper_author, pair,\
                filter_author_paper)
        features[i][9] = count
        if len(paper_author[pair[1]]) > 1:
            features[i][10] = count*1.0/len(paper_author[pair[1]])
        else:
            features[i][10] = -1
        # 9. number of co-author three year before current year
        count = make_year_coauthor_feature(-3, author_paper_year, paper_author, pair,\
                filter_author_paper)
        features[i][11] = count
        if len(paper_author[pair[1]]) > 1:
            features[i][12] = count*1.0/len(paper_author[pair[1]])
        else:
            features[i][12] = -1
        # 10. number of co-author five year before current year
        count = make_year_coauthor_feature(-5, author_paper_year, paper_author, pair,\
                filter_author_paper)
        features[i][13] = count
        if len(paper_author[pair[1]]) > 1:
            features[i][14] = count*1.0/len(paper_author[pair[1]])
        else:
            features[i][14] = -1

        (count, max_length, min_length) = make_length_year_feature(author_paper_year,\
                paper_author, pair, filter_author_paper)
        # 15. total count of years co-authored
        features[i][15] = count
        # 16. average total count of years co-authored
        if count == -1 or len(paper_author[pair[1]]) == 1:
            features[i][16] = -1
        else:
            features[i][16] = count*1.0 / (len(paper_author[pair[1]]) - 1)
        # 17. longest year
        features[i][17] = max_length
        # 18. shorted year
        features[i][18] = min_length


    return features

def find_first_year(year_list):
    for entry in year_list:
        if type(entry[0]) == int and entry[0] >  1900 and\
                entry[0] < 2014:
            return entry[0]
    return -1

def find_minmax_year_cnt(year_cnt):
    min_year_cnt= 10000
    min_year = -1
    max_year_cnt = 0
    max_year = -1
    for entry in year_cnt:
        if type(entry[0]) == int and entry[0] > 1900 and \
                entry[0] < 2014:
            if entry[1] < min_year_cnt:
                min_year_cnt = entry[1]
                min_year = entry[0]
            if entry[1] > max_year_cnt:
                max_year_cnt = entry[1]
                max_year = entry[0]
    return (max_year, min_year)

def write_features(features, feature_file):
    writer = csv.writer(open(feature_file, 'w'), lineterminator="\n")
    writer.writerows(features)

def make_year_coauthor_feature(year_range, author_paper_year, paper_author, pair,\
        filter_author_paper):
    # find papers that author published within a year_range (<0)
    if pair[0] not in author_paper_year:
        print 'author key error.'
        sys.exit(0)
    target_year = author_paper_year[pair[0]][pair[1]]
    target_authors = paper_author[pair[1]]

    # filter papers list accroding to their publication year
    paper_list = set()
    for entry in author_paper_year[pair[0]].items():
        if year_range >= 0:
            if entry[1] > 500 and abs(entry[1] - target_year) <= year_range\
                    and entry[0] in filter_author_paper[pair[0]]:
                paper_list.add(entry[0])
        else:
            if entry[1] > 500 and (entry[1] - target_year) > year_range\
                    and (entry[1] - target_year) <= 0 and entry[0] in\
                    filter_author_paper[pair[0]]:
                paper_list.add(entry[0])

    # count time-dependent co-author relationship
    num_coauthor = 0
    for target_author in target_authors:
        for paper in paper_list:
            if target_author != pair[0] and target_author in paper_author[paper]:
                num_coauthor += 1

    return num_coauthor

def make_length_year_feature(author_paper_year, paper_author, pair, filter_author_paper):
    length_year_coauthor = [0 for i in range(len(paper_author[pair[1]]))]
    target_year = author_paper_year[pair[0]][pair[1]]

    if target_year > 1900 and target_year <= 2013:
        global_min_length = 150
        for i, author in enumerate(paper_author[pair[1]]):
            max_length = 0
            for entry in author_paper_year[pair[0]].items():
                if author != pair[0]:
                    if entry[0] in filter_author_paper[pair[0]] and author in paper_author[entry[0]]:
                        length_year = target_year - author_paper_year[pair[0]][entry[0]]
                        if length_year > max_length:
                            max_length = length_year
                        if length_year >= 0 and length_year < global_min_length:
                            global_min_length = length_year
            length_year_coauthor[i] = max_length
        count = sum(length_year_coauthor)
        global_max_length = max(length_year_coauthor)
        return (count, global_max_length, global_min_length)
    else:
        return (-1, -1, -1)

# main function
def main():
    # reading paths
    print "Reading file path from the configuration file 'SETTINGS.json'"
    paths = json.loads(open("SETTINGS.json").read())
    postr_doc = paths["postr_doc"]
    negtr_doc = paths["negtr_doc"]
    vali_doc  = paths["vali_doc"]

    # loading instances
    print "Loading labeled instances."
    postr_ins = [ins for ins in csv.reader(open(postr_doc))]
    negtr_ins = [ins for ins in csv.reader(open(negtr_doc))]
    vali_ins = [ins for ins in csv.reader(open(vali_doc))]
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

    ## Make title and keyword similarity features
    # randomly partition users
    '''authors = list(set([entry[0] for entry in postr_ins]))
    random.shuffle(authors)
    init_set = authors[0:int(len(authors)*2.0/5)]
    train_set = authors[int(len(authors)*2.0/5)+1:int(len(authors)*4.5/5)]
    test_set = authors[int(len(authors)*4.5/5)+1:len(authors)]

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
    '''

    ## Make time dependent features
    # (1). time gap between the starting publition time of author and
    #    time of this paper;
    # (2). number of papers the author published in the same year as
    #    target paper;
    # (3). precentage of papers publised in the same year as target paper;
    # (4). time gap between the year of most publised paper and the year of
    #    target paper;
    # (5). time gap between the year of least publised paper and the year of
    #    target paper;
    # (6). time dependent co-author relationship
    # (7). year count features
    (author_paper_year, author_year_cnt, paper_author, filter_author_paper)=\
            load_year_info(paths['author_paper_year'],\
            paths['author_year_cnt'], paths['filter_author_paper'])

    postr_year_fea = make_year_feature(author_paper_year, author_year_cnt,\
            paper_author, filter_author_paper, [entry[0:2] for entry in postr_ins])
    features = [feature1 + feature2 for feature1, feature2 in zip(postr_ins, postr_year_fea)]
    write_features(features, paths["new_postr_doc"])
    del postr_year_fea

    negtr_year_fea = make_year_feature(author_paper_year, author_year_cnt,\
            paper_author, filter_author_paper, [entry[0:2] for entry in negtr_ins])
    features = [feature1 + feature2 for feature1, feature2 in zip(negtr_ins, negtr_year_fea)]
    write_features(features, paths["new_negtr_doc"])
    del negtr_year_fea

    vali_year_fea = make_year_feature(author_paper_year, author_year_cnt,\
            paper_author, filter_author_paper, [entry[0:2] for entry in vali_ins])
    features = [feature1 + feature2 for feature1, feature2 in zip(vali_ins, vali_year_fea)]
    write_features(features, paths["new_vali_doc"])
    del vali_year_fea


if __name__ == "__main__":
    main()
