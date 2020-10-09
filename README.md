# Disaster Response Pipeline Project
### Project Summary
Task of this project was to read in textdata that could possibly contain information on disasters (like medical help, water, ...). 
1st step: The dataset is cleaned (duplicates, etc.) and natural language processing in terms of tokenization (word_tokenize, lemmatize) is carried out to generate certain features. The dataset is saved in an SQL .db file.

2nd step: A Machine Learning pipeline with GridSearch included is created and a DecisionTree Classifer is trained on the trainset and evaluated on testset data. The trained model is then saved to a pickle file. The trained model will be used for further input of textdata in the app.

3rd step: An app is deployed (locally, but heroku or anything comparable would be possible as well), that shows the counts of messages by certain criteria and the total amount of counts of "Basic needs" in the dataset.



### Installation
Python Version >= 3.5x

Packages used:
	- numpy
    - pandas
    - regex
    - nltk
    - scikit-learn
    - plotly
    - flask
    - json
    - sqlalchemy
    - pickle

Dataset is used from [Figure-Eight](https://www.figure-eight.com/)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Descriptions:
- process_data.py is used to wrangle & clean the data. In addition it saves the cleaned data into a SQL .db file
- train-classifer.py is used to train and evaluate a Machine Learning Pipeline. The model is saved into a pickle file
- app.py is used for app deployment (locally or in the web) and visualization of specific data by Plotly (package)

For further details all functions provide a __doc__ method that explains the purpose
