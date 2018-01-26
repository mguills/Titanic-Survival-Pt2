# -*- coding: utf-8 -*-
"""
Clean up Titanic files
"""

import pandas as pd

def process_file(infilename, outfilename,
                 add_survival=False):
    # Open up the csv file in to a Python object
    df = pd.read_csv(infilename, header=0)
    df.info()
    
    if add_survival:
        names = map(lambda name: name.replace('"', ''), df_all.name)
        survived_dict = dict(zip(names, df_all.survived))
        cols = ['Survived'] + list(df.keys())
        df['Survived'] = df['Name'].map(lambda name: survived_dict[name.replace('"', '')])
        df = df.reindex(columns=cols)
    
    # Remap sex to binary value
    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Remap embarked to integer value
    df = df[df['Embarked'].notnull()]
    df['Embarked'] = df['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)
    
    # Drop features
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Filter missing values
    #for key in df.keys():
    #    df = df[df[key].notnull()]
    df = df.dropna()
    df.info()
    
    # Write out to file
    df.to_csv(outfilename, index=False)

df_all = pd.read_excel('../titanic_data/titanic3.xls', header=0)

process_file('../titanic_data/kaggle/train.csv', '../data/titanic_train.csv')
process_file('../titanic_data/kaggle/test.csv', '../data/titanic_test.csv')
process_file('../titanic_data/kaggle/test.csv', '../data/titanic_test.answers.csv', add_survival=True)