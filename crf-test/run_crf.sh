#!/bin/sh
prefix="../output/"
echo $prefix
crf_learn -t template "${prefix}trainingNegatedDataCRF.txt" "${prefix}model_negated"
crf_test  -m "${prefix}model_negated" "${prefix}testNegatedDataCRF.txt" > "${prefix}test_negated_out.csv"

echo "Negated concepts"
python count_out.py "${prefix}test_negated_out.csv"
python extract_concepts_found.py "${prefix}test_negated_out.csv" "${prefix}concepts_negated.csv"