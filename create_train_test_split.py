# Given a CSV with columns text and class,
# This script splits up the dataset into a train and test split

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import pdb

def main(data_csv, output_train, output_test, train_split=0.75):
    '''
        data_csv: path to data csv with two columns, text and class
        output_train: path to non-existent file, which will be created
        output_test: path to non-existent file, which will be created
        train_split: float, between 0 and 1, the fraction in the train
    '''
    df = pd.read_csv(data_csv, sep=',')
    X = df['text']
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=1-train_split, 
                                        random_state=42)
    # write train file 
    df = pd.DataFrame()
    df['text'] = X_train
    df['class'] = y_train    
    df.to_csv(output_train, quoting=1, index=False)
    # write test file
    df = pd.DataFrame()
    df['text'] = X_test
    df['class'] = y_test    
    df.to_csv(output_test, quoting=1, index=False)
    return 

if __name__ == '__main__':
    args = sys.argv[1:]
    data_csv = args[0]
    output_train = args[1]
    output_test = args[2]
    main(data_csv, output_train, output_test)
