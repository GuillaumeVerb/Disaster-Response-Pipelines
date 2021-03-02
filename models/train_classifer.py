# import packages
import sys
import pandas as pd
import pickle
import sqlite3
from sqlalchemy import create_engine
import re

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from statistics import mean

from util import Tokenizer, StartingVerbExtractor

def load_data(data_file):
    
    """
    Load SQLite database file to prepare input and labels
    """
    
    # read in file
    engine = create_engine( 'sqlite:///{}'.format(data_file) )
    df = pd.read_sql_table('rawdata',con=engine)

    # define features and label arrays
    X = df['message']
    Y = df.drop(labels=['id','message','original','genre'], axis=1)
    Y_col = Y.columns.values

    return X, Y, Y_col


def tokenize(text):
    '''
    Function to clean the text data  and apply tokenize and lemmatizer function
    Return the clean tokens
    Input: text
    Output: cleaned tokenized text as a list object
    '''

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ',text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()


    # Remove stop words
    # tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='n').strip()
        #I passed in the output from the previous noun lemmatization step. This way of chaining procedures is very common.
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        
        clean_tokens.append(clean_tok)
    
    
    return clean_tokens



def build_model():
    '''
    Function to build a model, create pipeline, hypertuning as well as gridsearchcv
    Input: N/A
    Output: Returns the model
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    parameters = {'tfidf__norm': ['l1','l2'],
              'clf__estimator__criterion': ["gini", "entropy"]    
             }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv




def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Function to evaluate a model and return the classificatio and accurancy score.
    Inputs: Model, X_test, y_test, category_names
    Outputs: Prints the Classification report & Accuracy Score
    '''

    y_pred = model.predict(X_test)
    report= classification_report(y_pred,Y_test, target_names=category_names)


    temp=[]
    for item in report.split("\n"):
        temp.append(item.strip().split('     '))
    clean_list=[ele for ele in temp if ele != ['']]
    report_df=pd.DataFrame(clean_list[1:],columns=['group','precision','recall', 'f1-score','support'])


    return report_df





def save_model(model, model_filepath):

    '''
    Function to save the model as pickle file in the directory
    Input: model and the file path to save the model
    Output: save the model as pickle file in the give filepath 
    '''

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
        
    """
    Classifier Main function
    
    This function implements Machine Learning Pipeline:
        1) Load data from SQLite db
        2) Train ML model on training set
        3) Evaluate model performance on test set
        4) Save trained model as Pickle
    
    """

    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')



if __name__ == '__main__':
    main()
