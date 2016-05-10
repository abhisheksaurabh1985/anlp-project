#!/bin/sh
crf_learn -c 5.0 -f 2 -t template trainingNegatedDataCRF.txt model_negated
crf_test  -m model_negated testNegatedDataCRF.txt > test_negated_out.csv

echo "Negated concepts"
python count_out.py test_negated_out.csv
python extract_concepts_found.py test_negated_out.csv concepts_negated.csv