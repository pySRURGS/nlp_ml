#/usr/bin/python
__doc__ = """
Natural language processing and machine learning by random search of the 
classifier and hyperparameter space.

Usage
-----
python nlp_ml.py train test iters path_to_db

Parameters 
----------
train: string
    An absolute or relative path to a CSV with two columns, first column name 
    must be 'text' and second column name must be 'class'. This dataset is used 
    to train the model.

test: string
    An absolute or relative path to a CSV with two columns, first column name 
    must be 'text' and second column name must be 'class'. This dataset is used 
    to test the model.

iters: int
    The number of different model/hyperparameter configurations to try out 

path_to_db: string
    An absolute or relative path to where we should save the database file        

Author
------
Sohrab Towfighi (C) 2019

License
-------
GPL version 3.0
https://www.gnu.org/licenses/gpl-3.0.en.html

Note
----
Developed using Python 3.7
"""
__version__ = '0.1'
import matplotlib
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_val_predict)
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb
import sys
import numpy as np
import argparse
import random
import sqlitedict
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from imblearn.over_sampling import SMOTE
from sqlitedict import SqliteDict
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin
import sklearn
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import tabulate
import spacy
import string
import warnings
warnings.filterwarnings("ignore")  # numpy 1.17 has issue with spacy

random.seed(0)
matplotlib.use("TkAgg")


class InvalidConfigError(Exception):
    pass


def spacy_tokenizer(sentence):
    punctuations = string.punctuation
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    parser = English()
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ !=
                '-PRON-' else word.lower_ for word in mytokens]
    mytokens = [
        word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens


def clean_text(text):
    return text.strip().lower()


class cleaner(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def generate_random_classifier():
    classifier_config_dict = {'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)},
        'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]},
        'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]},
        'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)},
        'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]},
        'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]},
        'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]},
        'MLPClassifier': {'hidden_layer_sizes': tuple([random.randint(1, 20) for x in range(0, random.randint(1, 20))]),
                          'solver': ['lbfgs']}
    }

    classifiers = list(classifier_config_dict.keys())
    chosen_clf = random.choices(classifiers)[0]
    arguments = list(classifier_config_dict[chosen_clf].keys())
    arguments_values = {}
    for arg in classifier_config_dict[chosen_clf].keys():
        arguments_values[arg] = random.choices(
            classifier_config_dict[chosen_clf][arg])[0]
    eval_str = chosen_clf + '('
    for arg in arguments_values.keys():
        value = arguments_values[arg]
        if type(value) == str:
            value = '"' + value + '"'
        else:
            value = str(value)
        eval_str = eval_str + arg + '=' + value + ','
    if eval_str.endswith(','):
        eval_str = eval_str[:-1]
    eval_str = eval_str + ')'
    clf = eval(eval_str)
    return clf


def generate_random_vectorizer():
    bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))
    tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)
    return random.choices([bow_vector, tfidf_vector])[0]


def load_X_y(path_to_csv):
    df = pd.read_csv(path_to_csv)
    X = df['text']
    y = df['class']
    X = X.tolist()
    return X, y


class CustomPipeline():
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier
        clean_transformer = cleaner()
        SMOTE_oversampler = SMOTE()
        self.cleaner = clean_transformer
        self.oversampler = SMOTE_oversampler
        self._train_accuracy = None
        self._train_precision = None
        self._train_recall = None
        self._test_accuracy = None
        self._test_precision = None
        self._test_recall = None

    def preprocess(self, X, y):
        X = self.cleaner.transform(X)
        X = self.vectorizer.fit_transform(X)
        X, y = self.oversampler.fit_resample(X, y)
        return X, y

    def cv_predict(self, X, y, store=True):
        try:
            y_pred = cross_val_predict(self.classifier,
                                       X, y, cv=10)
        except Exception as e:
            if ('Unsupported set of arguments:' in str(e) or
                    'Expected n_neighbors <= n_samples' in str(e)):
                raise InvalidConfigError
            else:
                raise e
        if store:
            self._train_accuracy = metrics.accuracy_score(y, y_pred)
            self._train_precision = metrics.precision_score(y, y_pred)
            self._train_recall = metrics.recall_score(y, y_pred)
        return y_pred

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def preprocess_predict(self, X, y):
        X_transform = self.cleaner.transform(X)
        X_transform = self.vectorizer.transform(X_transform)
        y_pred = self.classifier.predict(X_transform)
        self._test_accuracy = metrics.accuracy_score(y, y_pred)
        self._test_precision = metrics.precision_score(y, y_pred)
        self._test_recall = metrics.recall_score(y, y_pred)
        print(self._train_accuracy, self._train_precision, self._train_recall)
        return y_pred

    def summarize(self):
        return [
            self._train_accuracy,
            self._train_precision,
            self._train_recall,
            self._test_accuracy,
            self._test_precision,
            self._test_recall]


class PipelineList():
    def __init__(self, path_to_db):
        self._results = []
        with SqliteDict(path_to_db, autocommit=True) as results_dict:
            for key in results_dict.keys():
                self._results.append(results_dict[key])

    def sort(self):
        self._results = sorted(self._results, key=lambda x: x._train_accuracy,
                               reverse=True)

    def print(self, top=5):
        table = []
        header = ["_train_accuracy", "_train_precision", "_train_recall",
                  "_test_accuracy", "_test_precision", "_test_recall"]
        num_eqn = int(np.min((top, len(self._results))))
        for i in range(0, num_eqn):
            row = self._results[i].summarize()
            table.append(row)
        table_string = tabulate.tabulate(table, headers=header)
        print(table_string)


def main(train, test, path_to_db):
    examined_one_configuration = False
    while examined_one_configuration == False:
        try:
            classifier = generate_random_classifier()
            vectorizer = generate_random_vectorizer()
            X_train, y_train = load_X_y(train)
            pipeline = CustomPipeline(vectorizer, classifier)
            X_train, y_train = pipeline.preprocess(X_train, y_train)
            pipeline.cv_predict(X_train, y_train)
            pipeline.fit(X_train, y_train)
            X_test, y_test = load_X_y(test)
            y_pred = pipeline.preprocess_predict(X_test, y_test)
            with SqliteDict(path_to_db, autocommit=True) as results_dict:
                results_dict[str(vectorizer) + str(classifier)] = pipeline
            examined_one_configuration = True
        except InvalidConfigError:
            return -1


def select_and_save_best_model(pipelines, train_accuracy_criterion=0.9,
                               test_accuracy_criterion=0.9):
    pipelines.sort()
    chosen_model = None
    for i in range(0,len(pipelines._results)):
        pipeline = pipelines._results[i]
        if pipeline._train_accuracy > train_accuracy_criterion:
            if pipeline._test_accuracy > test_accuracy_criterion:
                chosen_model = pipeline
    if chosen_model is None:
        print("No model meets the accuracy criteria")
    else:
        with SqliteDict(path_to_db, autocommit=True) as results_dict:
            results_dict['best_model'] = chosen_model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='nlp_ml.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "train",
        help="absolute or relative file path to the training data csv")
    parser.add_argument(
        "test",
        help="absolute or relative file path to the testing data csv")
    parser.add_argument(
        "iters",
        help="the number of classifiers to be attempted in this run",
        type=int)
    parser.add_argument(
        "path_to_db",
        help="absolute or relative file path to the output sqlite database")
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    arguments = parser.parse_args()
    train = arguments.train
    test = arguments.test
    iters = arguments.iters
    path_to_db = arguments.path_to_db
    for i in range(0, iters):
        main(train, test, path_to_db)
    pipelines = PipelineList(path_to_db)
    pipelines.sort()
    pipelines.print()
    select_and_save_best_model(pipelines)
