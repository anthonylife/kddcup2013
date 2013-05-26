#!/usr/bin/env python
#encoding=utf8

import csv, json
from collections import defaultdict

def load_info(paper_year_conf_jour_csv, author_paper_csv):
    paper_year_conf_jour = defaultdict(list)
    instances = [ins for ins in csv.reader(open(paper_year_conf_jour_csv))]
    for entry in instances[1:]:
        paper_year_conf_jour[entry[0]] = [int(entry[1]), entry[2], entry[3]]

    author_paper = defaultdict(set)
    for entry in csv.reader(open(author_paper_csv)):
        author_paper[entry[0]].add(entry[1])
    return paper_year_conf_jour, author_paper

def make_time_confjour_features(paper_year_conf_jour, author_paper, pairs):
    num_features = 10
    features = [[-1000 for i in range(num_features)] for i in range(len(pairs))]

    for i, pair in enumerate(pairs):
        print i
        year_confjour_cnt = {}
        papers = author_paper[pair[0]]
        target_year = paper_year_conf_jour[pair[1]][0]
        target_conf = paper_year_conf_jour[pair[1]][1]
        target_jour = paper_year_conf_jour[pair[1]][2]

        for paper in papers:
            if paper not in paper_year_conf_jour:
                continue
            if (target_conf>0 and paper_year_conf_jour[paper][1]==target_conf) or\
                (target_jour>0 and paper_year_conf_jour[paper][2]==target_jour):
                    year = paper_year_conf_jour[paper][0]
                    if year > 1900 and year < 2014:
                        if year in year_confjour_cnt:
                            year_confjour_cnt[year] += 1
                        else:
                            year_confjour_cnt[year] = 1

        if len(year_confjour_cnt) > 0:
            year_cnt = sorted(year_confjour_cnt.items(), key = lambda x: x[0])
            # 1.first year the author published paper in conf or jour
            features[i][0] = year_cnt[0][0]
            if target_year > 1900 and target_year < 2014:
                # 2.year difference between first year the author published
                #   paper in conf or jour and current year
                features[i][1] = target_year - year_cnt[0][0]

            year_cnt = sorted(year_confjour_cnt.items(), key = lambda x: x[1])
            # 3.year the author published most paper in target conf or jour
            features[i][2] = year_cnt[0][0]
            if target_year > 1900 and target_year < 2014:
                # 4.year difference between the author published most paper
                #   in target conf or jour and current year
                features[i][3] = target_year - year_cnt[0][0]

            totalnum = sum([entry[1] for entry in year_cnt])
            if target_year > 1900 and target_year < 2014:
                pre_year = target_year - 1
                if pre_year in year_confjour_cnt:
                    # 5. number of papers published in previous year
                    features[i][4] = year_confjour_cnt[pre_year]
                    # 6. corresponding ratio
                    features[i][5] = year_confjour_cnt[pre_year]*1.0/totalnum
                else:
                    features[i][4] =0
                    features[i][5] =0

                if target_year in year_confjour_cnt:
                    # 7. number of papers published in current year
                    features[i][6] = year_confjour_cnt[target_year]
                    # 8. corresponding ratio
                    features[i][7] = year_confjour_cnt[target_year]*1.0/totalnum
                else:
                    features[i][6] =0
                    features[i][7] =0

                next_year = target_year + 1
                if next_year in year_confjour_cnt:
                    # 9. number of papers published in next year
                    features[i][8] = year_confjour_cnt[next_year]
                    # 10. corresponding ratio
                    features[i][9] = year_confjour_cnt[next_year]*1.0/totalnum
                else:
                    features[i][8] =0
                    features[i][9] =0

    return features

def write_features(features, feature_file):
    writer = csv.writer(open(feature_file, 'w'), lineterminator="\n")
    writer.writerows(features)


# main function
def main():
    # reading paths
    print "Reading file path from the configuration file 'SETTINGS.json'"
    paths = json.loads(open("SETTINGS.json").read())
    tc_csv = paths['basic_confirmed_features']
    td_csv = paths['basic_deleted_features']
    v_csv  = paths['basic_vali_features']


    print "Loading infomation which will be used to generate features later"
    (paper_year_conf_jour, author_paper) = \
            load_info(paths["paper_year_conf_jour"], paths["author_paper_year"])

    features = [entry[0:7] for entry in csv.reader(open(tc_csv))]
    time_additional_features = make_time_confjour_features(paper_year_conf_jour, author_paper, [entry[0:2] for entry in features])
    features = [feature1 + feature2 for feature1, feature2 in zip(features, time_additional_features)]
    write_features(features, paths["new_postr_doc"])

    features = [entry[0:7] for entry in csv.reader(open(td_csv))]
    time_additional_features = make_time_confjour_features(paper_year_conf_jour, author_paper, [entry[0:2] for entry in features])
    features = [feature1 + feature2 for feature1, feature2 in zip(features, time_additional_features)]
    write_features(features, paths["new_negtr_doc"])

    features = [entry[0:7] for entry in csv.reader(open(v_csv))]
    time_additional_features = make_time_confjour_features(paper_year_conf_jour, author_paper, [entry[0:2] for entry in features])
    features = [feature1 + feature2 for feature1, feature2 in zip(features, time_additional_features)]
    write_features(features, paths["new_vali_doc"])


if __name__ == "__main__":
    main()
