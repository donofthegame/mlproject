import os
import sys

import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
import pickle 
from sklearn.metrics import r2_score
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, Y_train, Y_test, models):
    try:
        logging.info("inside evaluate models function")
        report={}

        for model_name, model in models.items():
            logging.info(f"training model {model_name}")
            model.fit(X_train, Y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(Y_train, y_train_pred)
            test_model_score = r2_score(Y_test,y_test_pred)
            report[model_name] = test_model_score
            
        return report
    except Exception as e:
        CustomException(e, sys) 