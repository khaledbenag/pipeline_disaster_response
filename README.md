# Disaster Response Pipeline Project
<img src = "figures/emergency.jpg?raw=true" >
In this project, data engineering skills are applied to analyze disaster data and build a model for an API that classifies real messages sent during disaster events. A machine learning pipeline for text processing and multi-output classification is developed to classify these events in order to send the messages to an appropriate emergency agency. Finally, a Flask application is designed to help an emergency worker to get classification results in multiple categories for a given message.

# Requirements
 * Flask==2.0.1
 * plotly==4.4.1
 * nltk==3.6.2
 * joblib==0.14.1
 * pandas==1.3.2
 * numpy==1.19.5
 * SQLAlchemy==1.4.23
 * scikit_learn==0.24.2
# Project tree
```bash

├───templates
│       go.html # classification result page of web app
│       master.html # main page of web app
│
├───data
│       DisasterResponse.db # database to save clean data to
│       disaster_categories.csv # data to process 
│       disaster_messages.csv # data to process 
│       process_data.py
│
├───figures
│       emergency.jpg # image for readme
│       disaster-response-project.png # result
│
├───models
│       classifier.pkl
│       train_classifier.py # saved model after training (not included due to large size)
│
└───notebooks
        ETL Pipeline Preparation.ipynb 
        ML Pipeline Preparation.ipynb
```
# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains and saves classifier, please go to models folder (in bash: cd models)
        `python train_classifier.py ../data/DisasterResponse.db classifier.dill`
        the model is saved in a dill format.

2. Run the following command in the app's directory to run your web app.
    `python app.py`

3. Go to http://0.0.0.0:3001/

# Results

<img src= "figures/disaster-response-project.png">

# Deployment
The Flask API is deployed into Heroku. Procfile is added to specify the web engine and the starting web script (app.py).
For deployment some tips are used to avoid some bugs:
* First the tokenizer function is wrraped into a package to be use the same instance by ML pipeline and the flask app.
* If the model is saved in a .pkl format, some information used while training of the pipeline are not saved, which generates `__main__ attribute ` bug while deployment. Here we used dill library to overcome this problem.

You can visit the project at: https://disasterregarding.herokuapp.com

# Resources
* Udacity advanced data science nanodegree
* The amazing FLask tutorial by Miguel Grinberg: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world-legacy
* heroku.com
* The pickle load bug discussion: https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules 
     
# Acknowledgments
I would thank #Udacity for the data science advanced nanodegree program.
- disaster data from Figure Eight (https://appen.com/)
