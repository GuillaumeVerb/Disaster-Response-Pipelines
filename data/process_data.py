import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    Load raw data and merge raw data into singel data frame for data cleaning etc.
    
    Arguments:
            messages_filepath:          Raw messages datapath 
            categories_filepath:        Raw labels datapath
    Returns:
            df: Merged data frame
    """
    
    # Load the data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id').reset_index(drop=True)
    
    return df
    
def clean_data(df):
    """
    Data cleaning function.
    1. Remove child_alone label since it only has one value
    2. Drop duplicates
    """
    
    # create a dataframe of the 36 individual category columns
    categories_split = df.categories.str.split(pat=";",n=-1, expand=True)
    
    # Use first row to get the column names
    row = categories_split.iloc[0]
    category_colnames = [name.split("-")[0] for name in row]
    categories_split.columns = category_colnames

    # Merge splitted categoreis with original categries dataframe to get id back
    df = pd.concat([df, categories_split], axis=1)
    df.drop(labels='categories', axis=1, inplace=True)
    
    # set each value to be the last character of the string
    for column in category_colnames:
        df[column] = df[column].apply(lambda x:x.split('-')[-1]).astype(int)
    
    # child_alone only have 1 value and it is droped here
    df.drop(labels=['child_alone'], axis=1, inplace=True)
    
    # Drop duplicates and keep the first appearance.
    df.drop_duplicates(keep='first', inplace=True)
    
    # This is binary classification problem. However there are 3 values in the target variable.
    # Label 2 only has around 120 and are dropped here.
    df = df.loc[df.related!=2]
    
    return df


def save_data(df, database_filename):

    """
    Save the data to SQLite database for NLP model and further processing
    Arguments:
            df:                       Merged data frame
            database_filename:        database file path
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('rawdata', engine, if_exists='replace',index=False)


def main():
    
    """
    Data Processing function
    
    This function:
        1) Extract rawdata from .csv files
        2) Data cleaning and pre-processing
        3) Data saved to SQLite database
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
