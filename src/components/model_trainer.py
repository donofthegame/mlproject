import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models , params

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config= ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split target variables for test and target")
            X_train, Y_train, X_test, Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            logging.info("Models dict set")
            model_report = evaluate_models(X_train=X_train, Y_train=Y_train, X_test= X_test, Y_test=Y_test, models=models,params=params)
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No Best model found")
            logging.info(f"Best found model on both training and testing dataset {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(Y_test,predicted)

            return r2_square

        except Exception as e:
            CustomException(e, sys)
