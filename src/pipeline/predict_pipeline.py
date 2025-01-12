import pandas as pd
import pickle

class PredictPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_model()

    def load_model(self):
        try:
            with open("artifacts/model.pkl", "rb") as model_file:
                self.model = pickle.load(model_file)
            with open("artifacts/preprocessor.pkl", "rb") as preprocessor_file:
                self.preprocessor = pickle.load(preprocessor_file)
            print("Model and Preprocessor Loaded Successfully")
        except Exception as e:
            raise Exception(f"Error loading model or preprocessor: {e}")

    def predict(self, data):
        try:
            print(f"Input DataFrame:\n{data}")
            transformed_data = self.preprocessor.transform(data)
            prediction = self.model.predict(transformed_data)
            return prediction
        except Exception as e:
            raise Exception(f"Error during prediction: {e}")

class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "Promo": [self.data["Promo"]],
                "DayOfWeek": [self.data["DayOfWeek"]],
                "CompetitionDistance": [self.data["CompetitionDistance"]],
                "Store": [self.data["Store"]],
                "Customers": [self.data["Customers"]],
                "Open": [self.data["Open"]],
                "StateHoliday": [self.data["StateHoliday"]],
                "SchoolHoliday": [self.data["SchoolHoliday"]],
                "StoreType": [self.data["StoreType"]],
                "Assortment": [self.data["Assortment"]],
                "CompetitionOpenSinceMonth": [self.data["CompetitionOpenSinceMonth"]],
                "CompetitionOpenSinceYear": [self.data["CompetitionOpenSinceYear"]],
                "Promo2": [self.data["Promo2"]],
                "Promo2SinceWeek": [self.data["Promo2SinceWeek"]],
                "Promo2SinceYear": [self.data["Promo2SinceYear"]],
                "PromoInterval": [self.data["PromoInterval"]],
                "WeekOfYear": [self.data["WeekOfYear"]],
                "CompetitionOpen": [self.data["CompetitionOpen"]],
                "Promo2Open": [self.data["Promo2Open"]],
                "IsPromo2Month": [self.data["IsPromo2Month"]],
                "Year": [self.data["Year"]],
                "Month": [self.data["Month"]],
                "Day": [self.data["Day"]],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise Exception(f"Error creating DataFrame: {e}")
