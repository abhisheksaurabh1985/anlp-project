#!/bin/sh
crf_learn -c 10.0 -a MIRA template train.csv model
crf_test  -m model test.csv > test_out.csv
python count_out.py