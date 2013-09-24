import sys
import csv

def main():
   subpath = sys.argv[1]
   anspath = sys.argv[2]
   key_rank_sub = readRanking(subpath)
   key_rank_ans = readRanking(anspath)
   MAP = calcMAP(key_rank_sub, key_rank_ans)
   print "MAP=" + str(MAP)

def calcMAP(key_rank_sub, key_rank_ans):
   sumAP = 0.0
   for key in key_rank_sub.keys():
      subrank = key_rank_sub[key]
      ansrank = key_rank_ans[key]
      sumAP += calcAP(subrank, ansrank)
   MAP = sumAP / float(len(key_rank_sub.keys()))
   return MAP

def calcAP(pred, ans):
   ansset = set(ans)
   predMatch = [0.0] * len(pred)
   for idx in range(len(pred)):
      if pred[idx] in ansset:
         predMatch[idx] = 1.0
         ansset.remove(pred[idx])
   for idx in range(len(predMatch)):
      if idx == 0:
         continue
      predMatch[idx] += predMatch[idx - 1]
   for idx in range(len(predMatch)):
      predMatch[idx] = float(predMatch[idx]) / float(idx + 1)

   pred_precision = {}
   for idx in range(len(pred)):
      if pred[idx] not in pred_precision:
         pred_precision[pred[idx]] = predMatch[idx]

   AP = 0.0
   for trueid in set(ans):
      AP += pred_precision[trueid]
   AP /= len(ans)
   return AP

def readRanking(path):
   key_rank = {}
   for row in csv.reader(open(path)):
      key = int(row[0])
      ids = map(int, row[1].split())
      key_rank[key] = ids
   return key_rank

#def readRanking(path):
#   keys = []
#   ranklist = []
#   for row in csv.reader(open(path)):
#      key = int(row[0])
#      ids = map(int, row[1].split())
#      keys.append(key)
#      ranklist.append(ids)
#   keys_ranklist = sorted(zip(keys, ranklist))
#   ranklist = [x[1] for x in keys_ranklist]
#   return ranklist

#def readRanking(path):
#   ranklist = []
#   for row in csv.reader(open(path)):
#      key = int(row[0])
#      ids = map(int, row[1].split())
#      ranklist.append(ids)
#   return ranklist

if __name__ == '__main__':
   main()
