#!/usr/bin/env python
#encoding=utf8

import csv, json
from collections import defaultdict


def load_coauthor_info(paper_author_doc):
    author_author_dict = defaultdict(set)
    paper_author_dict  = defaultdict(set)

    init_paperid = '0'
    author_set = set()
    for entry in csv.reader(open(paper_author_doc)):
        if entry[0] == init_paperid:
            author_set.add(entry[1])
        else:
            for i, authorid in enumerate(author_set):
                for j, coauthorid in enumerate(author_set):
                    if i != j:
                        author_author_dict[authorid].add(coauthorid)
            author_set.clear()
            init_paperid = entry[0]
        paper_author_dict[entry[0]].add(entry[1])

    return author_author_dict, paper_author_dict


def mk_twohop_coauthor(author_author_dict, paper_author_dict, pairs):
    features = [[-1, -1] for i in range(len(pairs))]
    for i, pair in enumerate(pairs):
        print i
        if len(paper_author_dict[pair[1]]) == 1:
            continue
        for authorid in paper_author_dict[pair[1]]:
            if authorid == pair[0]:
                continue
            features[i][0] +=\
                    len(author_author_dict[authorid]&author_author_dict[pair[0]])
        features[i][1] = features[i][0]*1.0/(len(paper_author_dict[pair[1]])-1)

    return features


def load_confjour_author_info(paper_confjour_doc, paper_author_dict):
    confjour_author_dict = defaultdict(set)

    for entry in csv.reader(open(paper_confjour_doc)):
        for authorid in paper_author_dict[entry[0]]:
            confjour_author_dict[entry[1]].add(authorid)

    return confjour_author_dict


def mk_twohop_confjour(paper_author_dict, confjour_author_dict, pairs):
    features = [[-1, -1] for i in range(len(pairs))]
    for i, pair in enumerate(pairs):
        for authorid in paper_author_dict[pair[1]]:
            if len(paper_author_dict[pair[1]]) == 1:
                continue
            for confid in confjour_author_dict.keys():
                if authorid == pair[0]:
                    continue
                if authorid in confjour_author_dict[confid] and\
                        pair[0] in confjour_author_dict[confid]:
                    features[i][0] += 1
            features[i][1] = features[i][0]*1.0/(len(paper_author_dict[pair[1]])-1)

    return features


def write_features(doc, features):
    writer = csv.writer(open(doc, 'w'), lineterminator="\n")
    writer.writerows(features)


def main():
    # reading paths
    print "Reading file path from the configuration file 'SETTINGS.json'"
    paths = json.loads(open("SETTINGS.json").read())
    tc_doc = paths["tc_9th_doc"]
    td_doc = paths["td_9th_doc"]
    v_doc  = paths["v_9th_doc"]

    # loading instances
    print "Loading labeled instances."
    tc_features = [ins[0:7] for ins in csv.reader(open(tc_doc))]
    td_features = [ins[0:7] for ins in csv.reader(open(td_doc))]
    v_features = [ins[0:7] for ins in csv.reader(open(v_doc))]

    # loading authors' co-author dictionary
    (author_author_dict, paper_author_dict) = load_coauthor_info(paths["paperauthor_doc"])
    features = mk_twohop_coauthor(author_author_dict, paper_author_dict,\
            [entry[0:2] for entry in tc_features])
    tc_features = [feature1+feature2 for feature1, feature2 in zip(tc_features, features)]
    features = mk_twohop_coauthor(author_author_dict, paper_author_dict,\
            [entry[0:2] for entry in td_features])
    td_features = [feature1+feature2 for feature1, feature2 in zip(td_features, features)]
    features = mk_twohop_coauthor(author_author_dict, paper_author_dict,\
            [entry[0:2] for entry in v_features])
    v_features = [feature1+feature2 for feature1, feature2 in zip(v_features, features)]
    del author_author_dict

    # loading authors' conf dictionary
    conf_author_dict = load_confjour_author_info(paths["paper_conf"], paper_author_dict)
    features = mk_twohop_confjour(paper_author_dict, conf_author_dict,\
            [entry[0:2] for entry in tc_features])
    tc_features = [feature1+feature2 for feature1, feature2 in zip(tc_features, features)]
    features = mk_twohop_confjour(paper_author_dict, conf_author_dict,\
            [entry[0:2] for entry in td_features])
    td_features = [feature1+feature2 for feature1, feature2 in zip(td_features, features)]
    features = mk_twohop_confjour(paper_author_dict, conf_author_dict,\
            [entry[0:2] for entry in v_features])
    v_features = [feature1+feature2 for feature1, feature2 in zip(v_features, features)]
    del conf_author_dict

    # loading authors' journal dictionary
    jour_author_dict = load_confjour_author_info(paths["paper_jour"], paper_author_dict)
    features = mk_twohop_confjour(paper_author_dict, jour_author_dict,\
            [entry[0:2] for entry in tc_features])
    tc_features = [feature1+feature2 for feature1, feature2 in zip(tc_features, features)]
    features = mk_twohop_confjour(paper_author_dict, jour_author_dict,\
            [entry[0:2] for entry in td_features])
    td_features = [feature1+feature2 for feature1, feature2 in zip(td_features, features)]
    features = mk_twohop_confjour(paper_author_dict, jour_author_dict,\
            [entry[0:2] for entry in v_features])
    v_features = [feature1+feature2 for feature1, feature2 in zip(v_features, features)]
    del jour_author_dict

    # write features
    tc_doc = paths["new_postr_doc"]
    td_doc = paths["new_negtr_doc"]
    v_doc  = paths["new_vali_doc"]
    write_features(tc_doc, tc_features)
    write_features(td_doc, td_features)
    write_features(v_doc, v_features)

if __name__ == "__main__":
    main()
