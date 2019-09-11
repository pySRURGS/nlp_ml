import os
import sys
import glob
import pdb
import nlp_ml
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

train = './spam_data/SMSSpamCollection_train.csv'
test = './spam_data/SMSSpamCollection_test.csv'
iters = 10
path_to_db = './output.db'
sh.python('nlp_ml.py', train, test, iters, path_to_db)
