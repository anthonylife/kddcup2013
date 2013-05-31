#!/bin/bash
#
# Copyright (c) 2013 group kddcup2013 project
# Author: anthonylife
# Date:   2013/5/25
# 
# Third party: RankLib-v2.1
#


# Global variable setting
CV_NUM=5
MODEL_IDX=0

CV_TC_SEPNUM=(0 24689 $[24689*2] $[24689*3] $[24689*4] $[123447])
CV_TD_SEPNUM=(0 22492 $[22492*2] $[22492*3] $[22492*4] $[112462])

# Feature construction
#python ensembleFormatFeature.py

## Note: modify the file name of different prediction results of the models
# Training and Prediction
tc_prob_file="../../ensembleData/tc_mart.txt"
td_prob_file="../../ensembleData/td_mart.txt"
v_prob_file="../../ensembleData/v_mart.txt"
#tc_prob_file="../../ensembleData/tc_rankboost.txt"
#td_prob_file="../../ensembleData/td_rankboost.txt"
#v_prob_file="../../ensembleData/v_rankboost.txt"
#tc_prob_file="../../ensembleData/tc_adarank.txt"
#td_prob_file="../../ensembleData/td_adarank.txt"
#v_prob_file="../../ensembleData/v_adarank.txt"
#tc_prob_file="../../ensembleData/tc_listnet.txt"
#td_prob_file="../../ensembleData/td_listnet.txt"
#v_prob_file="../../ensembleData/v_listnet.txt"

touch $tc_prob_file
touch $td_prob_file
touch $v_prob_file
for ((i=0; i<$CV_NUM; ++i))
do
    cmd=$[${CV_TC_SEPNUM[$i]}+1]","${CV_TC_SEPNUM[$i+1]}"p"
    sed -n $cmd ../../features/tc_ltr.txt > ./tc_feature.txt
    cmd=$[${CV_TD_SEPNUM[$i]}+1]","${CV_TD_SEPNUM[$i+1]}"p"
    sed -n $cmd ../../features/td_ltr.txt > ./td_feature.txt
    
    if (( i == 0));
    then
        cmd1=$[${CV_TC_SEPNUM[$i+1]}+1]","${CV_TC_SEPNUM[$CV_NUM]}"p"
        sed -n $cmd1 ../../features/tc_ltr.txt > ./trainfeature.txt
        cmd1=$[${CV_TD_SEPNUM[$i+1]}+1]","${CV_TD_SEPNUM[$CV_NUM]}"p"
        sed -n $cmd1 ../../features/td_ltr.txt >> ./trainfeature.txt
    elif (( i == 4 )); 
    then
        cmd1=$[${CV_TC_SEPNUM[0]}+1]","${CV_TC_SEPNUM[$CV_NUM-1]}"p"
        sed -n $cmd1 ../../features/tc_ltr.txt > ./trainfeature.txt
        cmd1=$[${CV_TD_SEPNUM[0]}+1]","${CV_TD_SEPNUM[$CV_NUM-1]}"p"
        sed -n $cmd1 ../../features/td_ltr.txt >> ./trainfeature.txt
    else
        cmd1=$[${CV_TC_SEPNUM[0]}+1]","${CV_TC_SEPNUM[$i-1]}"p"
        sed -n $cmd1 ../../features/tc_ltr.txt > ./trainfeature.txt
        cmd1=$[${CV_TD_SEPNUM[0]}+1]","${CV_TD_SEPNUM[$i-1]}"p"
        sed -n $cmd1 ../../features/td_ltr.txt >> ./trainfeature.txt
        cmd1=$[${CV_TC_SEPNUM[$i+1]}+1]","${CV_TC_SEPNUM[$CV_NUM]}"p"
        sed -n $cmd1 ../../features/tc_ltr.txt >> ./trainfeature.txt
        cmd1=$[${CV_TD_SEPNUM[$i+1]}+1]","${CV_TD_SEPNUM[$CV_NUM]}"p"
        sed -n $cmd1 ../../features/td_ltr.txt >> ./trainfeature.txt
    fi
    echo $[$i+1]"th iteration..." 
    read haha
    # Model training
    java -jar RankLib-v2.1/bin/RankLib.jar -train ./trainfeature.txt -ranker $MODEL_IDX -metric2t MAP -metric2T MAP -save model.dat

    # Model testing
    java -jar RankLib-v2.1/bin/RankLib.jar -load model.dat -rank ./tc_feature.txt -metric2T MAP -score test.prediction
    cat test.prediction >> $tc_prob_file

    java -jar RankLib-v2.1/bin/RankLib.jar -load model.dat -rank ./td_feature.txt -metric2T MAP -score test.prediction
    cat test.prediction >> $td_prob_file
done

cat ../../features/tc_ltr.txt > ./trainfeature.txt
cat ../../features/td_ltr.txt >> ./trainfeature.txt
java -jar RankLib-v2.1/bin/RankLib.jar -train ./trainfeature.txt -ranker $MODEL_IDX -metric2t MAP -metric2T MAP -save model.dat
java -jar RankLib-v2.1/bin/RankLib.jar -load model.dat -rank ./td_feature.txt -metric2T MAP -score test.prediction
cat test.prediction > $v_prob_file

# Clear
rm ./tc_feature.txt
rm ./td_feature.txt
rm ./trainfeature.txt
rm ./test.prediction
