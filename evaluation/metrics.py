#!/usr/bin/env python
#encoding=utf8

import numpy as np
from sklearn.metrics import average_precision_score

def eval_map(author_predictions):
    ''' Input:
            # author_predictions: a dict where author is the key and has 3 attributes
                --> 1.prob; 2.paperid; 3.label.
        Output:
            # MAP value.
    '''
    sum_ap = 0.0
    for author_id in author_predictions:
        paperid_set = set([])
        scores_list = []
        label_list = []
        for entry in author_predictions[author_id]:
            if entry[1] not in paperid_set:
                paperid_set.add(entry[1])
                scores_list.append(entry[0])
                label_list.append(entry[2])
            #else:
                #scores_list.append(entry[0])
                #label_list.append(0)
        y_scores = np.array(scores_list)
        y_label  = np.array(label_list)
        if 0 not in set(label_list):
            sum_ap += 1
        else:
            sum_ap += average_precision_score(y_label, y_scores)

    return sum_ap*1.0 / len(author_predictions)


def eval_map_recovery(author_predictions):
    sum_ap = 0.0
    for author_id in author_predictions:
        paperid_repeated_set = set([])
        precision = 0.0
        right_num = 0
        sorted_author_papers = sorted(author_predictions[author_id],\
                key=lambda x:x[0], reverse=True)
        for i, entry in enumerate(sorted_author_papers):
            if entry[1] not in paperid_repeated_set:
                if entry[2] == 1:
                    right_num += 1
                    precision += right_num*1.0/(i+1)
                    paperid_repeated_set.add(entry[1])
        sum_ap += precision / right_num

    return sum_ap*1.0 / len(author_predictions)
