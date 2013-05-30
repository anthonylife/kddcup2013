import csv, json
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB


MODEL_NUM = 3

CV_NUM = 5
CV_TC_SEPNUM = [0, 24689, 24689*2, 24689*3, 24689*4, 123447]
CV_TD_SEPNUM = [0, 22492, 22492*2, 22492*3, 22492*4, 112462]

def main():
    print("Loading paths")
    paths = json.loads(open("SETTINGS.json").read())

    print("Getting features for deleted papers from the disk file")
    features_conf = [feature for feature in csv.reader(open(paths["trainpos_features"]))]
    features_deleted = [feature for feature in csv.reader(open(paths["trainneg_features"]))]

    td_features = [map(lambda y: float(y), x[2:]) for x in features_deleted]
    tc_features = [map(lambda y: float(y), x[2:]) for x in features_conf]

    td_target = [0 for x in range(len(features_deleted))]
    tc_target = [1 for x in range(len(features_conf))]

    print("Training the Classifier")

    ## Model choices
    if MODEL_NUM == 1:
        # Logistic Regression
        classifier = LogisticRegression(C=0.1, dual=False, fit_intercept=True,\
                intercept_scaling=1, penalty='l2', tol=0.0001)
        tc_wfd = open(paths['tc_logistic'], 'w')
        td_wfd = open(paths['td_logistic'], 'w')
        v_wfd = open(paths['v_logistic'], 'w')
    elif MODEL_NUM == 2:
        # LinearSVC
        classifier = LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0,\
                multi_class='ovr',fit_intercept=True, intercept_scaling=1, class_weight=None,\
                verbose=0, random_state=None)
        tc_wfd = open(paths['tc_linearSVC'], 'w')
        td_wfd = open(paths['td_linearSVC'], 'w')
        v_wfd = open(paths['v_linearSVC'], 'w')
    elif MODEL_NUM == 3:
        # BernoulliNB
        classifier = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
        tc_wfd = open(paths['tc_BNB'], 'w')
        td_wfd = open(paths['td_BNB'], 'w')
        v_wfd = open(paths['v_BNB'], 'w')
    else:
        sys.exit(1)

    # training and prediction
    tc_probs = [0.0 for i in range(len(tc_features))]
    td_probs = [0.0 for i in range(len(td_features))]

    for i in range(CV_NUM):
        print("crossvalidation num: #%d" % (i+1))
        classifier.fit(tc_features[0:CV_TC_SEPNUM[i]] + tc_features[CV_TC_SEPNUM[i+1]+1:] +\
                td_features[0:CV_TD_SEPNUM[i]] + td_features[CV_TD_SEPNUM[i+1]+1:],\
                tc_target[0:CV_TC_SEPNUM[i]] + tc_target[CV_TC_SEPNUM[i+1]+1:] +\
                td_target[0:CV_TD_SEPNUM[i]] + td_target[CV_TD_SEPNUM[i+1]+1:])

        if MODEL_NUM == 2:
            predictions = classifier.predict(tc_features[CV_TC_SEPNUM[i]:CV_TC_SEPNUM[i+1]])
        else:
            predictions = classifier.predict_proba(tc_features[CV_TC_SEPNUM[i]:CV_TC_SEPNUM[i+1]])[:,1]
        predictions = list(predictions)
        tc_probs[CV_TC_SEPNUM[i]:CV_TC_SEPNUM[i+1]] = predictions

        if MODEL_NUM == 2:
            predictions = classifier.predict(td_features[CV_TD_SEPNUM[i]:CV_TD_SEPNUM[i+1]])
        else:
            predictions = classifier.predict_proba(td_features[CV_TD_SEPNUM[i]:CV_TD_SEPNUM[i+1]])[:,1]
        predictions = list(predictions)
        td_probs[CV_TD_SEPNUM[i]:CV_TD_SEPNUM[i+1]] = predictions

    tc_wfd.writelines(["%s\n" % prob for prob in tc_probs])
    tc_wfd.close()
    td_wfd.writelines(["%s\n" % prob for prob in td_probs])
    td_wfd.close()

    # clear memcache
    del td_probs, predictions

    features_vali = [map(lambda y: float(y), feature[2:]) for feature in csv.reader(open(paths['vali_features']))]
    classifier.fit(tc_features+td_features, tc_target+td_target)
    if MODEL_NUM == 2:
        v_probs = classifier.predict(features_vali)
    else:
        v_probs = classifier.predict_proba(features_vali)[:,1]
    v_wfd.writelines(["%s\n" % prob for prob in v_probs])
    v_wfd.close()

if __name__=="__main__":
    main()
