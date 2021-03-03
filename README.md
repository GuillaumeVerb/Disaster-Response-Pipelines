
# Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.</br>

This project is divided in the following key sections:</br>

  - Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB</br>
  - Build a machine learning pipeline to train the which can classify text message in various categories</br>
  - Run a web app which can show model results in real time</br>

# Installing

Clone this GIT repository: </br>

git clone https://github.com/GuillaumeVerb/Disaster-Response-Pipelines.git

# Executing Program:
Run the following commands in the project's root directory to set up your database and model.</br>

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db</br>
To run ML pipeline that trains classifier and saves python models/train_classifer.py data/DisasterResponse.db models/classifier.pkl</br>
Run the following command in the app's directory to run your web app. python run.py</br>

Go to http://0.0.0.0:3001/


# Disaster-Response-Pipelines


- app </br>
| - template</br>
| |- master.html  # main page of web app</br>
| |- go.html  # classification result page of web app</br>
|- run.py  # Flask file that runs app</br>

- data</br>
|- categories.csv  # data to process </br>
|- messages.csv  # data to process</br>
|- process_data.py</br>
|- DisasterResponse.db   # database to save clean data to</br>

- models</br>
|- train_classifer.py</br>
|- classifier.pkl  # saved model </br>

- README.md


# Acknowledgements
Udacity for providing an amazing Data Science Nanodegree Program</br>
Figure Eight for providing the relevant dataset to train the model</br>

