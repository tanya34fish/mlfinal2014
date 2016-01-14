#!/bin/bash
python hog.py ml14fall_train.dat train_hog_feature.txt
python hog.py ml14fall_test1_no_answer.dat test1_hog_feature.txt
g++ -o convert.out convert.c
./convert.out train_hog_feature.txt train_hog_feature_sparse.txt
./convert.out test1_hog_feature.txt test1_hog_feature_sparse.txt
./svm-train -t 0 train_hog_feature_sparse.txt hog_feature.model
./svm-predict test1_hog_feature_sparse.txt hog_feature.model test1_hog_feature.predict
