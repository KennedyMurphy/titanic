# This script adapts the data cleansing and feature
# engineering of the following kernel:
# https://www.kaggle.com/jeetranjeet619/titanic-dataset-play-around-in-r/code
import logging
import numpy as np
import pandas as pd
from itertools import product

logger = logging.getLogger(__name__)

def clean_dataset(df):
    """ Given the full data frame of the titanic dataset,
        replaces missing values with the appropriate
        value.

        :param df:      data frame of full titanic data set
        
        :return:        Cleansed data set
    """

    logger.info("Cleaning titanic data set")

    logger.info("Dropping columns with 75% or more missing values")
    for col in df.columns:
        na_count = df[col].isnull().sum()

        if na_count / len(df) >= 0.75:
            logger.debug("Dropping column {} with {}% missing vals".format(
                col, round(na_count/len(df) * 100, 2)))
            df.drop(columns=col, inplace=True)

    ### Fare ###
    combos =  product(df['Pclass'].unique(), df['Embarked'].unique())
    for pclass, embarked in combos:

        sub = df[
            (df['Pclass'] == pclass) &
            (df['Embarked'] == embarked)
        ]

        if sub['Fare'].isnull().any():
            logger.debug(
                "Replacing missing values for fare of Pclass: {} Embarked: {}".format(
                    pclass, embarked))
            
            df.loc[sub['Fare'].isnull().index, 'Fare'] = sub['Fare'].median()
    
    ### Age ###
    if df['Age'].isnull().any():
        na_count = df['Age'].isnull().sum()

        logger.debug("{} of {} Age values are missing. Replacing with median.".format(
            na_count, len(df)))
        
        df.loc[df['Age'].isnull(), 'Age'] = df['Age'].median()
    
    ### Embarked ###
    # in Embarked, the rows are not exactly marked as NA but instead 
    # some have few empty strings so just checking for NA wont give you anything.
    
    # There are two entries with missing embarked flags. Both with
    # fare = 80, Pclass = 1, Sex = Female. Observing the distributions
    # for simular observations shows that S is more appropriate. E.g. the
    # mean/median fare for Pclass 1 Female passengers are closest to 80.
    # See the kernel mentioned above for details.
    df.loc[(df['Embarked'].isnull()) | (df['Embarked'] == ""), 'Embarked'] == 'S'

    return df


def build_features(df):
    """ Extracts the features from the cleaned data set.

        :param df:      Cleaned data frame

        :return:        Feature data frame.
    """

    # Start the feature df
    feats = df[['PassengerId', 'Survived', 'train_set']].copy()

    logger.info("Creating title features")
    df['Title'] = df['Name'].apply(parse_title)
    df['active'] = 1

    title_df = df.set_index(['PassengerId', 'Title'])['active'].unstack()
    title_df.fillna(0, inplace=True)

    feats = feats.merge(title_df, on='PassengerId', how='left')

    del title_df

    logger.info("Creating family size features")
    # Alternative: Create bucketed one-hot features for family size.
    fsize = df[['PassengerId', 'SibSp', 'Parch', "Age", "Sex", 'Title']].copy()
    fsize['Fsize'] = fsize[['SibSp', 'Parch']].sum(axis=1) + 1 # Include themselves

    fsize['Mother'] = np.where((fsize['Sex'] == 'female') & 
                                (fsize['Age'] > 18) & 
                                (fsize['Parch'] > 0) &
                                (fsize['Title'] != 'Miss'),
                                1, 0)

    feats = feats.merge(fsize[['PassengerId', 'Fsize', 'Mother']], 
                        on='PassengerId', how='left')
    feats['Solo'] = np.where(feats['Fsize'] == 1, 1, 0)

    del fsize

    logger.info("Creating age and sex group features")
    age_df = df[['PassengerId', 'Age', 'Sex']].copy()
    age_df['Female'] = np.where(age_df['Sex'] == 'female', 1, 0)

    age_df['Infant'] = np.where(age_df['Age'] <= 4, 1, 0)
    age_df['Child'] = np.where((age_df['Age'] > 4) & (age_df['Age'] <= 10), 1, 0)
    age_df['Young'] = np.where((age_df['Age'] > 10) & (age_df['Age'] <= 18), 1, 0)
    age_df['Adult'] = np.where((age_df['Age'] > 18) & (age_df['Age'] <= 50), 1, 0)
    age_df.drop(columns=['Age', 'Sex'], inplace=True)

    feats = feats.merge(age_df, on='PassengerId', how='left')

    del age_df

    logger.info("Creating fare range features")
    fare_df = df[['PassengerId', 'Fare', 'Embarked']].copy()
    fare_df['Embarked_S'] = np.where(df['Embarked'] == 'S', 1, 0)
    fare_df['Embarked_C'] = np.where(df['Embarked'] == 'C', 1, 0)

    fare_df.drop(columns='Embarked', inplace=True)
    feats = feats.merge(fare_df, on='PassengerId', how='left')

    return feats


def parse_title(x):
    """ Extracts the title (Mr., Mrs., etc.) from
        the provided name.

        :param x:   Name to parse
        
        :return:    title
    """

    titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.']

    # If we are able to find the title in the name, return it.
    for t in titles:
        if t in x:
            return t.replace(".", "")
    
    # Else return Rare for the rare title.
    return "Rare"


def main():
    
    logger.info("Reading in training/test set.")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train['train_set'] = True
    test['train_set'] = False

    df = pd.concat([train, test], sort=True)

    del train, test

    df = clean_dataset(df)
    df = build_features(df)

    train = df[df['train_set']]
    test = df[~df['train_set']]

    train.drop(columns='train_set', inplace=True)
    test.drop(columns='train_set', inplace=True)

    train.to_csv("data/train_features.csv", index=False)
    test.to_csv("data/test_features.csv", index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    main()