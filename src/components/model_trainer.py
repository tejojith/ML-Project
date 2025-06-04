import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Splitting training and testing input and target feature")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1], #take out the last column
                train_array[:,-1], 
                test_array[:,:-1],  
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Liner Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "XGB Classifier": XGBRegressor(),
                "CatBoost Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict = evaluate_model(x_train = x_train,
                                               y_train = y_train,
                                               x_test = x_test,
                                               y_test = y_test,
                                               models = models)
            
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            
            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with score: {best_model_score}")    

            save_object(
                file_path = self.trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            return CustomException(e, sys)
 