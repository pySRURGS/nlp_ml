import os
import sys
import glob
import pdb
import nlp_ml
from nlp_ml import CustomPipeline, spacy_tokenizer, cleaner
import numpy as np
from sqlitedict import SqliteDict
try:
    import sh
except ImportError:
    # fallback: emulate the sh API with pbs
    import pbs

    class Sh(object):
        def __getattr__(self, attr):
            return pbs.Command(attr)
    sh = Sh()

train = './spam_data/SMSSpamCollectionMini_train.csv'
test = './spam_data/SMSSpamCollectionMini_test.csv'
iters = 2
path_to_db = './output.db'
sh.python('nlp_ml.py', train, test, iters, path_to_db)
