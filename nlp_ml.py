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
from mlxtend.evaluate import confusion_matrix
from plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import pdb
import sys
import numpy as np
import argparse
import random
import sqlitedict
import pandas as pd
from imblearn.over_sampling import SMOTE
from sqlitedict import SqliteDict
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin
from sklearn.utils.testing import all_estimators
import sklearn
import scikitplot as skplt
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import tabulate
import spacy
import string
import warnings
warnings.filterwarnings("ignore")  # numpy 1.17 has issue with spacy

random.seed(0)

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
    classifier_config_dict = {
        'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False],
        'verbose': [3]},
        'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'verbose': [3]},
        'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'verbose': [3]},
        'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False],
        'verbose': [3]},
        'MLPClassifier': {
        'hidden_layer_sizes': tuple([random.randint(1, 20) for x
                                     in range(0, random.randint(1, 20))]),
        'solver': ['lbfgs'], 
        'verbose': [True]}
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


def find_classifiers_with_predict_proba():
    estimators = all_estimators()
    estimators_with_predict_proba = []
    for name, class_ in estimators:
        if hasattr(class_, 'predict_proba'):
            estimators_with_predict_proba.append(name)
    return estimators_with_predict_proba


def generate_random_vectorizer():
    bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))
    tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)
    return random.choices([bow_vector, tfidf_vector])[0]


def load_X_y(path_to_csv):
    df = pd.read_csv(path_to_csv)
    X = df['text']
    y = df['class']
    X = np.array(X)
    return X, y

def load_X(path_to_csv):
    df = pd.read_csv(path_to_csv)
    X = df['text']
    X = X.tolist()
    return X, y    

class CustomPipeline():
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self._estimator_type = 'classifier'
        self.classifier = classifier
        clean_transformer = cleaner()
        SMOTE_oversampler = SMOTE()
        self.cleaner = clean_transformer
        self.oversampler = SMOTE_oversampler
        self._train_accuracy = None
        self._train_precision = None
        self._train_recall = None
        self._train_roc_auc = None
        self._test_accuracy = None
        self._test_precision = None
        self._test_recall = None
        self._test_roc_auc = None

    def fit_preprocess(self, X, y, oversample=True):
        X = self.cleaner.transform(X)
        X = self.vectorizer.fit_transform(X)
        if oversample:
            try:
                X, y = self.oversampler.fit_resample(X, y)
            except ValueError:
                pass
        return X, y

    def cv_predict(self, X, y, store=True):
        X, y = self.fit_preprocess(X, y)
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
            self._train_roc_auc = metrics.roc_auc_score(y, y_pred)
            self._train_precision = metrics.precision_score(y, y_pred)
            self._train_recall = metrics.recall_score(y, y_pred)
        return y_pred

    def fit(self, X, y):
        X_transform, y = self.fit_preprocess(X, y)
        self.classifier.fit(X_transform, y)

    def predict_proba(self, X):
        X_transform = self.preprocess(X)
        y_pred = self.classifier.predict_proba(X_transform)
        return y_pred

    def predict(self, X):
        X_transform = self.preprocess(X)
        y_pred = self.classifier.predict(X_transform)
        return y_pred

    def preprocess(self, X):
        X_transform = self.cleaner.transform(X)
        X_transform = self.vectorizer.transform(X_transform)
        return X_transform

    def preprocess_predict(self, X):
        X_transform = self.preprocess(X)
        y_pred = self.classifier.predict(X_transform)
        return y_pred

    def calculate_performance(self, X, y):
        y_pred = self.preprocess_predict(X)
        self._test_accuracy = metrics.accuracy_score(y, y_pred)
        self._test_precision = metrics.precision_score(y, y_pred)
        self._test_recall = metrics.recall_score(y, y_pred)
        self._test_roc_auc = metrics.roc_auc_score(y, y_pred)

    def summarize(self):
        return [
            self._train_accuracy,
            self._train_precision,
            self._train_recall,
            self._train_roc_auc,
            self._test_accuracy,
            self._test_precision,
            self._test_recall,
            self._test_roc_auc]


class PipelineList():
    def __init__(self, path_to_db):
        self._results = []
        self._best_pipeline = None
        with SqliteDict(path_to_db, autocommit=True) as results_dict:
            for key in results_dict.keys():
                self._results.append(results_dict[key])
            try:
                self._best_pipeline = results_dict['best_model']
            except:
                pass

    def sort(self):
        self._results = sorted(self._results, key=lambda x: x._train_accuracy,
                               reverse=True)

    def print(self, top=5):
        table = []
        header = ["_train_accuracy", "_train_precision", "_train_recall", 
                  "_train_roc_auc", "_test_accuracy", "_test_precision", 
                  "_test_recall", "_test_roc_auc"]
        num_eqn = int(np.min((top, len(self._results))))
        for i in range(0, num_eqn):
            row = self._results[i].summarize()
            table.append(row)
        table_string = tabulate.tabulate(table, headers=header, floatfmt=".2f")        
        print(table_string)
        return table_string


def run(train, test, path_to_db):
    examined_one_configuration = False
    while examined_one_configuration == False:
        try:
            classifier = generate_random_classifier()
            vectorizer = generate_random_vectorizer()
            with SqliteDict(path_to_db, autocommit=True) as results_dict:
                try:
                    results_dict[str(vectorizer) + str(classifier)]
                    continue
                except KeyError:
                    pass
            X_train, y_train = load_X_y(train)
            pipeline = CustomPipeline(vectorizer, classifier)
            pipeline.cv_predict(X_train, y_train)
            pipeline.fit(X_train, y_train)
            X_test, y_test = load_X_y(test)
            pipeline.calculate_performance(X_test, y_test)
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
        pipelines._best_pipeline = chosen_model        
    return chosen_model

def make_plots(train, test, pipelines):
    extensions = ['svg', 'eps', 'png']
    X_train, y_train = load_X_y(train)
    X_test, y_test = load_X_y(test)
    pipelines.sort()
    clf = pipelines._results[0]
    y_pred = clf.predict(X_test)
    classifiers_with_predict_proba = find_classifiers_with_predict_proba()
    if clf.classifier.__class__.__name__ in classifiers_with_predict_proba:
        y_probas = clf.predict_proba(X_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_probas)
        fig = plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='best')        
        fig.set_size_inches(4,3)
        for ext in extensions:
            plt.savefig("./figures/roc_curve."+ext)
    else:
        print(clf.classifier.__class__.__name__,"not in predict proba list")
    cm = confusion_matrix(y_target=y_test, 
                          y_predicted=y_pred, 
                          binary=True)
    plot_confusion_matrix(conf_mat=cm, colorbar=True)
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.tight_layout()
    for ext in extensions:
        plt.savefig("./figures/confusion_matrix."+ext)
    plt.clf()
    '''
    plot_learning_curves(X_train, y_train, X_test, y_test, 
                         clf, style='seaborn-colorblind', print_model=False)
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.ylabel("Misclassification fraction")
    plt.tight_layout()
    for ext in extensions:
        plt.savefig("./figures/learning_curve."+ext)
    '''

def main(train, test, iters, path_to_db, predict):
    if predict is not None:
        X = load_X(predict)
        pipelines = PipelineList(path_to_db)
        best_pipe = select_and_save_best_model(pipelines)
        y_pred = best_pipe.preprocess_predict(X)
        print(y_pred)
        exit(0)
    for i in range(0, iters):
        run(train, test, path_to_db)
    pipelines = PipelineList(path_to_db)
    pipelines.sort()
    table = pipelines.print()
    with open("./figures/table.csv", "w") as text_file:
        text_file.write(table)
    select_and_save_best_model(pipelines)
    make_plots(train, test, pipelines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='nlp_ml.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "train",
        help="absolute or relative file path to the training data CSV")
    parser.add_argument(
        "test",
        help="absolute or relative file path to the testing data CSV")
    parser.add_argument(
        "iters",
        help="the number of classifiers to be attempted in this run",
        type=int)
    parser.add_argument(
        "path_to_db",
        help="absolute or relative file path to the output sqlite database")
    parser.add_argument(
        "-predict",
        help="absolute or relative file path to a CSV; will predict labels for CSV based on best model from database; `test` and `train` get ignored")
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    arguments = parser.parse_args()
    train = arguments.train
    test = arguments.test
    iters = arguments.iters
    path_to_db = arguments.path_to_db
    predict = arguments.predict
    main(train, test, iters, path_to_db, predict)
