#!/usr/bin/env python
#encoding=utf8

import csv, re, sys
sys.path.append('../')
from nltk.stem import PorterStemmer
from toolkit.filterwords import Stopwords

# create word map for titles and keywords
def create_map(src_file, target_file, field_id):
    stemmer = PorterStemmer()
    stopwords = Stopwords()
    file_content = [row for row in csv.reader(open(src_file))]

    simple_content = symreplace(file_content, field_id)
    clean_dict = preprocess(simple_content, stemmer, stopwords)

    wordlist = clean_dict.items()
    wordlist = sorted(wordlist, key = lambda x:x[0])
    wfd = open(target_file, 'w')
    for word in wordlist:
        wfd.write("%s, %d\n" % (word[0], word[1]))
    wfd.close()


# replace some symbols which may cause ambiguilty
def symreplace(in_content, field_id):
    pattern = re.compile(r'\W')
    if field_id == 5:
        pattern_sup = re.compile(r'\w.* \w.*:')

    out_content = []
    for line in in_content:
        pro_line = line[field_id].lower()
        if field_id == 5:
            pro_line = pattern_sup.sub('', pro_line)
        pro_line = pattern.sub(' ', pro_line)
        out_content.append(pro_line)

    return out_content

# convert words to lower case and do stemming
def preprocess(raw_str, stemmer, stopwords):
    clean_dict = {}
    temp_dict = {}
    pattern = re.compile(r'.*\d.*')

    linenum = 0
    for line in raw_str:
        linenum += 1
        print linenum
        fragments = line.lower().split(' ')
        for fragment in fragments:
            if len(fragment) > 1:
                stemmed_word = fragment
                #stemmed_word = stemmer.stem_word(fragment)
                if stemmed_word not in temp_dict and \
                        not stopwords.is_stopword(fragment):
                    temp_dict[stemmed_word] = 1
                elif stemmed_word in temp_dict and \
                        not stopwords.is_stopword(fragment):
                    temp_dict[stemmed_word] = temp_dict[stemmed_word] + 1
    for key in temp_dict.keys():
        if temp_dict[key] > 10 and not pattern.match(key):
            clean_dict[key] = len(clean_dict)

    print len(clean_dict)
    return clean_dict


def main(paperinfo, titlemap, keywdmap):
    # create word map for titles
    create_map(paperinfo, titlemap, 1)
    # create word map for keywords
    create_map(paperinfo, keywdmap, 5)


if __name__ == "__main__":
    paperinfo = "../../data/Paper.csv"
    titlemap  = "../../dict/Titlemap"
    keywdmap  = "../../dict/Keywdmap"

    main(paperinfo, titlemap, keywdmap)
