import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Reads in two csv files and merges them on column "id"
    
<<<<<<< HEAD
    Inputs: str / path to csvfiles
    Returns: pd.DataFrame / df
=======
    Inputs: str. path to csvfiles
    Returns: pd.DataFrame. df
>>>>>>> e451611e4fb36aed7ccee285d9e7cdaec5b213ff
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on = 'id', how = 'outer')
    
    return df


def clean_data(df):
    '''
    Creates a cleaned dataframe by 
     - splitting strings, 
     - applying column names to categories,
     - dropping "Old" labels,
     - concatting cleaned and original data
     - removing duplicates
    
    Input: pd.DataFrame / df
    Returns: pd.DataFrame / df
    '''
    
    # create a dataframe of the N individual category columns
    categories = df['categories'].str.split(";", expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # Remove the last two strings to get a colname
    category_colnames = row.apply(lambda x: x[0:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(labels = ['categories'], axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # check number of duplicates
    print('Number of duplicates before cleaning: ', df.duplicated().sum())
    # drop duplicates
    df = df.drop_duplicates(keep = 'first')
    # check number of duplicates
    print('Number of duplicates after cleaning: ', df.duplicated().sum())
   
    return df

def save_data(df, database_filename):
    '''
    saves df to a sql database filen
    
    Input: 
        pd.DataFrame / df
        str / database_filename
    Returns:
        -
    '''
<<<<<<< HEAD
=======
    
>>>>>>> e451611e4fb36aed7ccee285d9e7cdaec5b213ff
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
<<<<<<< HEAD
       # 'disaster_messages.csv', 'disaster_categories.csv', 'disaster_01' = sys.argv[1:]
=======
>>>>>>> e451611e4fb36aed7ccee285d9e7cdaec5b213ff

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
    
#cmd command line
#python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db