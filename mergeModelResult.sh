#!/bin/bash

# counting the number of adopted models
model_num=7
tc_model_result=[./ensemble/tc_linearSVC.txt, ./ensemble/tc_logistic.txt, ./ensemble/tc_BNB.txt, ./learningToRank/tc_mart.txt, ./learningToRank/tc_rankboost.txt, ./learningToRank/adarank.txt, ./learningToRank/listnet.txt]
td_model_result=[./ensemble/td_linearSVC.txt, ./ensemble/td_logistic.txt, ./ensemble/td_BNB.txt, ./learningToRank/tc_mart.txt, ./learningToRank/tc_rankboost.txt, ./learningToRank/adarank.txt, ./learningToRank/listnet.txt]
tc_model_result=[./ensemble/tc_linearSVC.txt, ./ensemble/tc_logistic.txt, ./ensemble/tc_BNB.txt, ./learningToRank/tc_mart.txt, ./learningToRank/tc_rankboost.txt, ./learningToRank/adarank.txt, ./learningToRank/listnet.txt]

count=0
cp ./data/tc_0.9726_abs.csv ./tc_temp0
for ((i=1; i<=$model_num; ++i))
do
    paste -d, ./tc_temp$[$i-1] ${tc_model_result[$i]} > ./tc_temp$i
    rm ./tc_temp$[$i-1]
    paste -d, ./td_temp$[$i-1] ${td_model_result[$i]} > ./td_temp$i
    rm ./td_temp$[$i-1]
    paste -d, ./v_temp$[$i-1] ${v_model_result[$i]} > ./v_temp$i
    rm ./v_temp$[$i-1]
done
mv ./tc_temp$model_num ./features/trainConfirmedFeatures.csv
mv ./td_temp$model_num ./features/trainDeletedFeatures.csv
mv ./v_temp$model_num ./features/validFeatures.csv
