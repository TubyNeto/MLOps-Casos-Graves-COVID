import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection._split import StratifiedKFold

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

import mlflow

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SelectFdr

import optuna
from sklearn.metrics import f1_score, precision_score, recall_score

@task
def data_preparation(path="C:/Users/Tuby Neto/Desktop/MLOps/sintomas_covid.csv"):
    dataset = pd.read_csv(path,index_col=0)
    def creating_age_groups(x):
        #Recem nascido
        if 0 <= x <= 5:
            x = '0 - 5'
        #crianças
        elif 6 <= x <= 15:
            x = '6 - 15'
        #Adolescente e jovens adultos
        elif 16 <= x <= 25:
            x = '16 - 25'
        #Adultos
        elif 26 <= x <= 40:
            x = '26 - 40'
        #Meia idade
        elif 41 <= x <= 60:
            x = '41 - 60'
        #Idosos
        elif 61 <= x <= 80:
            x = '61 - 80'
        #idosos com mais de 80 anos
        else:
            x = '>80'
        
        return x

    dataset['age_group'] = dataset['idade']
    dataset['age_group'] = dataset['age_group'].apply(creating_age_groups)

    dataset = pd.get_dummies(dataset, columns = ['age_group'])
    dataset = dataset.drop("idade",axis=1)

    return dataset

@task
def train_models(dataset):

    X = dataset.drop("death",axis=1)
    y = dataset['death']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    

    mlflow.sklearn.autolog()
    with mlflow.start_run():

        mlflow.set_tag("model", "Random Forest")
        
        mlflow.log_param("train-data-path", pd.concat([X_train,y_train],axis=1))
        mlflow.log_param("valid-data-path", pd.concat([X_valid,y_valid],axis=1))

        best_params={'bootstrap': True, 
                    'criterion': 'entropy', 
                    'max_features': 'log2', 
                    'min_samples_leaf': 1, 
                    'min_samples_split': 6, 
                    'n_estimators': 1973}
        
        mlflow.log_param("alpha", best_params)
        
        rf = RandomForestClassifier(**best_params)
        rf.fit(X_train, y_train)

        with open('C:/Users/Tuby Neto/Desktop/MLOps/random_forest.bin', 'wb') as f_out:
            pickle.dump(rf, f_out)

        y_pred = rf.predict(X_valid)
        f1 = f1_score(y_valid,y_pred)
        
        mlflow.log_metric("f1", f1)

        mlflow.log_artifact(local_path="C:/Users/Tuby Neto/Desktop/MLOps/random_forest.bin", artifact_path="models_pickle")


    
    
    mlflow.sklearn.autolog()
    with mlflow.start_run():

        mlflow.set_tag("model", "Naive Bayes")
        
        mlflow.log_param("train-data-path", pd.concat([X_train,y_train],axis=1))
        mlflow.log_param("valid-data-path", pd.concat([X_valid,y_valid],axis=1))

        best_params={'fit_prior': True, 
                    'alpha': 0.5655935981286497, 
                    'norm': False}
        
        mlflow.log_param("alpha", best_params)
        
        cnb = ComplementNB(**best_params)
        cnb.fit(X_train, y_train)

        with open('C:/Users/Tuby Neto/Desktop/MLOps/naive_bayes.bin', 'wb') as f_out:
            pickle.dump(cnb, f_out)

        y_pred = cnb.predict(X_valid)
        f1 = f1_score(y_valid,y_pred)
        
        mlflow.log_metric("f1", f1)

        mlflow.log_artifact(local_path="C:/Users/Tuby Neto/Desktop/MLOps/naive_bayes.bin", artifact_path="models_pickle")

    return

@flow(task_runner=SequentialTaskRunner())
def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("projeto-mlops-experiment")
    dataset = data_preparation()
    train_models(dataset)

main()
