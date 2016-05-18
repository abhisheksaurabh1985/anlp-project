ConceptSentimentClassifier
==========================

- Authors: Ilmira Terpugova, Abhishek Saurabh, Mohammed Bashir Kazimi, Andrei Polzounov
- Date: 18/05/2016
- Version: 1.0


The system is created to classify the concept inside the sentence into two type of classes" "Negated" and "Affirmed".

The model uses crf++ implementation
https://taku910.github.io/crfpp/

We used CRF++-0.58.tar.gz in our experiments.
Python 2.7 is used.

The dataset is provided in /data folder.

The method uses nested kfold cross-validation, the outer k = 5, the inner k = 10.

 execute.py - main file to train the model and evaluate results;
 negclassifier_functions.py - file with function defenitions;
 negclassifier_classes.py - file with classes definitions;
 data/Annotations-1-120_orig.txt - file for training
 data/negex_triggers.txt - file with pre- and postnegation cues
 templates - folder for the features templates
 output - folder where all output files are placed