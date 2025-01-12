import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for imputing missing values in the 'CompetitionDistance' column
    based on the provided logic.
    """
    def __init__(self, column_name):
        self.column_name = column_name
        self.max_distance = None

    def fit(self, X, y=None):
        # Calculate the maximum value for the column
        self.max_distance = X[self.column_name].max()
        return self

    def transform(self, X):
        # Apply the imputation logic
        X = X.copy()
        X[self.column_name] = X[self.column_name].fillna(self.max_distance * 2)

        return X

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        """
        try:
            numerical_columns = ['Store', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 
                                 'CompetitionOpen', 'Promo2', 'Promo2Open', 'IsPromo2Month',
                                 'Day', 'Month', 'Year', 'WeekOfYear']
            categorical_columns = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment']
            
            num_pipeline = Pipeline(
                steps=[
                    ("custom_imputer", CustomImputer(column_name="CompetitionDistance")),
                    
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Sales"
            numerical_columns = ['Store', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 
                                 'CompetitionOpen', 'Promo2', 'Promo2Open', 'IsPromo2Month',
                                 'Day', 'Month', 'Year', 'WeekOfYear']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
