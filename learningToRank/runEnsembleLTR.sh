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
MODEL_IDX=0     #-----Modification------#

CV_TC_SEPNUM=(0 24689 $[24689*2] $[24689*3] $[24689*4] 123447)
CV_TD_SEPNUM=(0 22492 $[22492*2] $[22492*3] $[22492*4] 112462)

# Feature construction
#python ensembleFormatFeature.py

## Note: modify the file name of different prediction results of the models
# Training and Prediction
tc_prob_file="../../ensembleData/tc_mart.txt"           #-----Modification------#
td_prob_file="../../ensembleData/td_mart.txt"
v_prob_file="../../ensembleData/v_mart.txt"
#tc_prob_file="../../ensembleData/tc_rankboost.txt"     #-----Modification------#
#td_prob_file="../../ensembleData/td_rankboost.txt"
#v_prob_file="../../ensembleData/v_rankboost.txt"
#tc_prob_file="../../ensembleData/tc_adarank.txt"       #-----Modification------#
#td_prob_file="../../ensembleData/td_adarank.txt"
#v_prob_file="../../ensembleData/v_adarank.txt"
#tc_prob_file="../../ensembleData/tc_listnet.txt"       #-----Modification------#
#td_prob_file="../../ensembleData/td_listnet.txt"
#v_prob_file="../../ensembleData/v_listnet.txt"


# input files
tc_ltr_file=../../features/tc_ltr.txt
td_ltr_file=../../features/td_ltr.txt 
v_ltr_file=../../features/v_ltr.txt 

# nomination of temporal files
tc_feature_file=./tc_mart_feature.txt                   #-----Modification------#
td_feature_file=./td_mart_feature.txt
train_feature_file=./trainfeature_mart.txt
model_file=./mart_model.dat
pre_result=./mart_prediction.txt
#tc_feature_file=./tc_rankboost_feature.txt             #-----Modification------#
#td_feature_file=./td_rankboost_feature.txt
#train_feature_file=./trainfeature_rankboost.txt
#model_file=./rankboost_model.dat
#pre_result=./rankboost_prediction.txt
#tc_feature_file=./tc_adarank_feature.txt               #-----Modification------#
#td_feature_file=./td_adarank_feature.txt
#train_feature_file=./trainfeature_adarank.txt
#model_file=./adarank_model.dat
#pre_result=./adarank_prediction.txt
#tc_feature_file=./tc_listnet_feature.txt               #-----Modification------#
#td_feature_file=./td_listnet_feature.txt
#train_feature_file=./trainfeature_listnet.txt
#model_file=./listnet_model.dat
#pre_result=./listnet_prediction.txt


# output files management 
rm $tc_prob_file
rm $td_prob_file
rm $v_prob_file
touch $tc_prob_file
touch $td_prob_file
touch $v_prob_file

for ((i=0; i<$CV_NUM; ++i))
do
    cmd=$[${CV_TC_SEPNUM[$i]}+1]","${CV_TC_SEPNUM[$i+1]}"p"
    sed -n $cmd $tc_ltr_file > $tc_feature_file
    cmd=$[${CV_TD_SEPNUM[$i]}+1]","${CV_TD_SEPNUM[$i+1]}"p"
    sed -n $cmd $td_ltr_file > $td_feature_file
    
    if (( i == 0));
    then
        cmd1=$[${CV_TC_SEPNUM[$i+1]}+1]","${CV_TC_SEPNUM[$CV_NUM]}"p"
        sed -n $cmd1 $tc_ltr_file > $train_feature_file
        cmd1=$[${CV_TD_SEPNUM[$i+1]}+1]","${CV_TD_SEPNUM[$CV_NUM]}"p"
        sed -n $cmd1 $td_ltr_file >> $train_feature_file
    elif (( i == 4 )); 
    then
        cmd1=$[${CV_TC_SEPNUM[0]}+1]","${CV_TC_SEPNUM[$CV_NUM-1]}"p"
        sed -n $cmd1 $tc_ltr_file > $train_feature_file
        cmd1=$[${CV_TD_SEPNUM[0]}+1]","${CV_TD_SEPNUM[$CV_NUM-1]}"p"
        sed -n $cmd1 $td_ltr_file >> $train_feature_file
    else
        cmd1=$[${CV_TC_SEPNUM[0]}+1]","${CV_TC_SEPNUM[$i-1]}"p"
        sed -n $cmd1 $tc_ltr_file > $train_feature_file
        cmd1=$[${CV_TD_SEPNUM[0]}+1]","${CV_TD_SEPNUM[$i-1]}"p"
        sed -n $cmd1 $td_ltr_file >> $train_feature_file
        cmd1=$[${CV_TC_SEPNUM[$i+1]}+1]","${CV_TC_SEPNUM[$CV_NUM]}"p"
        sed -n $cmd1 $tc_ltr_file >> $train_feature_file
        cmd1=$[${CV_TD_SEPNUM[$i+1]}+1]","${CV_TD_SEPNUM[$CV_NUM]}"p"
        sed -n $cmd1 $td_ltr_file >> $train_feature_file
    fi
    
    echo
    echo $[$i+1]"th iteration..." 
    
    # Model training
    java -jar RankLib-v2.1/bin/RankLib.jar -train $train_feature_file -ranker $MODEL_IDX -metric2t MAP -metric2T MAP -save $model_file

    # Model testing
    java -jar RankLib-v2.1/bin/RankLib.jar -load $model_file -rank $tc_feature_file -metric2T MAP -score $pre_result
    cat $pre_result >> $tc_prob_file

    java -jar RankLib-v2.1/bin/RankLib.jar -load $model_file -rank $td_feature_file -metric2T MAP -score $pre_result
    cat $pre_result >> $td_prob_file

done

cat $tc_ltr_file > $train_feature_file
cat $td_ltr_file >> $train_feature_file
java -jar RankLib-v2.1/bin/RankLib.jar -train $train_feature_file -ranker $MODEL_IDX -metric2t MAP -metric2T MAP -save $model_file
java -jar RankLib-v2.1/bin/RankLib.jar -load $model_file -rank $v_ltr_file -metric2T MAP -score $pre_result
cat $pre_result > $v_prob_file

# Clear
rm $tc_feature_file
rm $td_feature_file
rm $train_feature_file
rm $model_file
rm $pre_result
