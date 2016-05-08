#!/bin/sh
crf_learn -c 20.0 -f 15 -t template trainingAffirmedDataCRF.txt model_affirmed
crf_test  -m model_affirmed testAffirmedDataCRF.txt > test_affirmed_out.csv

crf_learn -c 20.0 -f 15 -t template trainingNegatedDataCRF.txt model_negated
crf_test  -m model_negated testNegatedDataCRF.txt > test_negated_out.csv

echo "Affirmed concepts"
python count_out.py test_affirmed_out.csv
python extract_concepts_found.py test_affirmed_out.csv concepts_affirmed.csv

echo "Negated concepts"
python count_out.py test_negated_out.csv
python extract_concepts_found.py test_negated_out.csv concepts_negated.csv