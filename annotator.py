#
# This script can be used when annotating the text data
# We start with a CSV without annotated labels, 'source'
# We add labels to this CSV but save it in a new file, 'target'
# 
import csv
import os 
import sys
import argparse
import pandas as pd
import pdb
try:
    import sh
except ImportError:
    # fallback: emulate the sh API with pbs
    import pbs
    class Sh(object):
        def __getattr__(self, attr):
            return pbs.Command(attr)
    sh = Sh()

def initialize_target(source, target):
    if os.path.isfile(target) == False:
        sh.cp(source, target)
    else:
        print(target, "exists, delete it manually if you are sure, then run me")

if __name__ == '__main__':
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog='annotator.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "source_filepath",
        help="absolute or relative file path to the source data CSV")
    parser.add_argument(
        "target_filepath",
        help="absolute or relative file path to the data CSV now with your labels")
    parser.add_argument(
        "labels_list",
        help="double semicolon delimited list of the header entries corresponding to classifications needing to be annotated")
    parser.add_argument(
        "text_column_header_value",
        help="the name of the column with text data for NLP")
    parser.add_argument(
        "-setup",
        help="should we create the target file from the source file. this should be run once when we start annotating.")
    arguments = parser.parse_args()
    source_filepath = arguments.source_filepath
    target_filepath = arguments.target_filepath
    labels_list = arguments.labels_list.split(';;')
    text_column_header_value = arguments.text_column_header_value
    if arguments.setup is not None:
        initialize_target(source_filepath, target_filepath)
    source = pd.read_csv(source_filepath)
    target = pd.read_csv(target_filepath)        
    for ind in target.index:
        print_text = False
        for label in labels_list:
            if target[label][ind] not in [0, 1, 2, '0', '1', '2', 0.0, 1.0, 2.0]:
                print(target[label][ind])
                print_text = True 
        if print_text == True:
            print('-------------------------------------')
            print('-------------------------------------')
            print(target[text_column_header_value][ind])
            for label in labels_list:
                print(label)
                print("Input 0, 1, 2 or `s` for skip, then press enter")
                if label_value.strip().lower() == 's'
                    label_value = input()
                target.loc[ind, label] = label_value.strip()            
        target.to_csv(target_filepath)        
    