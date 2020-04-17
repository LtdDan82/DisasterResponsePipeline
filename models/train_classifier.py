import sys

# import libraries
import pandas as pd
import re
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

#%%
def load_data(database_filepath):
    '''
    Loads data from sql database
    
    Input: str / database_filepath
    Returns: 
        pd.Series / X
        pd.DataFrame / Y
        list / category_names
    '''
<<<<<<< HEAD
    
    engine = create_engine('sqlite:///' + database_filepath)
=======
    engine = create_engine('sqlite:///' + database_filepath)
#    filepath = database_filepath.rsplit('/')[1]
>>>>>>> e451611e4fb36aed7ccee285d9e7cdaec5b213ff
    df = pd.read_sql(database_filepath, con = engine)
    X = df['message']
    Y = df.loc[:, ~df.columns.isin(['message', 'id', 'original', 'genre'])]
    category_names = Y.columns
    
    return X, Y, category_names
<<<<<<< HEAD

=======
#%%
>>>>>>> e451611e4fb36aed7ccee285d9e7cdaec5b213ff
def tokenize(text):
    '''
    Tokenizes a string object:
        - all lowercase
        - word tokenization
        - stop word removal
        - (stemming)
        - Lemmatization
        
    Input: str / text
    Returns: str / text
    '''
<<<<<<< HEAD
    
=======
>>>>>>> e451611e4fb36aed7ccee285d9e7cdaec5b213ff
    #Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize
    text = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    text = [w for w in text if not w in stop_words]
    #text = [PorterStemmer().stem(w) for w in text]
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
     
    return text


def build_model():
    '''
    builds a model pipeline:
        - tfidf (Vectorizer)
        - classifier
        
    Input: -
    Returns: sklearn.model object / model
    '''
<<<<<<< HEAD
    
=======
>>>>>>> e451611e4fb36aed7ccee285d9e7cdaec5b213ff
    pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer = tokenize)),
                    ('clf', MultiOutputClassifier(DecisionTreeClassifier()))])
    
    parameters = {'clf__estimator__max_depth': [3, 9],
             'clf__estimator__min_samples_split': [2, 5]}
    
    cv = GridSearchCV(pipeline, param_grid = parameters)
    model = cv
    
    return model
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the testset and creates a classification report for every column
    
    Input:
        sklearn.model object / model
        pd.DataFrame / X_test, Y_test
        list / category_names
    Returns: -
    '''
    
    y_pred = model.predict(X_test)
    #Generate a pd.DataFrame form y_pred
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    for col in Y_test.columns:
        print(classification_report(Y_test[col], y_pred[col]))


def save_model(model, model_filepath):
    '''
    saves the model to a pickle file
    
    Input: 
        sklearn.model object / model
        str / model_filepath
    Returns:
        -
    '''
<<<<<<< HEAD
    
=======
>>>>>>> e451611e4fb36aed7ccee285d9e7cdaec5b213ff
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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