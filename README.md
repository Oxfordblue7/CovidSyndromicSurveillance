# covidSyndromicSurveillance
 A preliminary system to automatically annotate COVID-related symptoms from social media posts.


## The main program

The main script of the program is **autoAnnotate.py**.


## Data

The data used for ***training*** is the annotation file **s4.xlsx**.

The data used for ***evaluating*** is the gold standard annotation file **Assignment1GoldStandardSet.xlsx**.

The data used for ***predicting*** is the unlabeled file **UnlabeledSet2.xlsx**.


## Outputs

All training and testing outputs are in the directory **outputs**.

The system output of annotations for the unlabeled set **UnlabeledSet2.xlsx** is **autoAnnotations_unlabeledSet2.xlsx**.

## New files

**neg_trigs2.txt** 
This file contains several more negation indicators than the original file **neg_trigs.txt**.

**COVID-Twitter-Symptom-Lexicon_stdSymptom.txt** 
This file uses the standard symptoms as the symptom expressions.

**COVID-Twitter-Symptom-Lexicon_s4labeled.txt** 
This files builds lexicons from the **s4.xlsx** annotations.

**COVID-Twitter-Symptom-Lexicon_new.txt** 
A combination of **COVID-Twitter-Symptom-Lexicon.txt** and **COVID-Twitter-Symptom-Lexicon_stdSymptom.txt**.

**COVID-Twitter-Symptom-Lexicon_new2.txt** 
A combination of **COVID-Twitter-Symptom-Lexicon.txt** and **COVID-Twitter-Symptom-Lexicon_s4labeled.txt**.
