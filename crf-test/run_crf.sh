#!/bin/sh
crf_learn -c 20.0 -f 16 template trainingDataCRF.txt model
crf_test  -m model testDataCRF.txt > test_out.csv
python count_out.py