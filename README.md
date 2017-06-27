README for Machine Learning NanoDegree Capstone Project
Peter Herr
May 23 2017
#############################

This project contains the following files:
1. Machine Learning Engineer Nanodegree - Capstone Project.pdf
2. MachineLearningNanodegree-CapstoneProposal_v2.pdf
3. Udacity Capstone Project - Final.ipynb
4. Data used for project (https://www.kaggle.com/c/quora-question-pairs/data), includes: train.csv, test.csv, and sample_submission.csv

Kaggle Quora Question Project URL: https://www.kaggle.com/c/quora-question-pairs

This project was coded in Python 2.7. The following python packages and libraries were used:
- import numpy as np
- import pandas as pd
- import matplotlib.pyplot as plt
- import seaborn as sns
- import time
- from __future__ import print_function, division
- from string import punctuation
- import re
- from nltk.stem import PorterStemmer
- import Levenshtein as lev
- from sklearn.feature_extraction.text import TfidfVectorizer
- from sklearn.metrics.pairwise import cosine_similarity
- from sklearn.cross_validation import train_test_split
- from sklearn.model_selection import GridSearchCV
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.metrics import log_loss 
- from sklearn.linear_model import LogisticRegression
- import xgboost as xgb
