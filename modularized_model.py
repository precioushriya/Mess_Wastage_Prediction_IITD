import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages')


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from typing import Dict
import numpy as np

# 1. Data Class for Input Data
@dataclass
class InputData:
    quantity_of_food: float
    number_of_students: int
    type_of_food: str
    day_of_week: str
    storage_conditions: str
    seasonality: str
    pricing: float

# 2. Preprocessing and Feature Engineering Class
class DataPreprocessor:
    def __init__(self):
        self.categorical_features = ['Type of Food', 'Day of the Week', 'Storage Conditions', 'Seasonality', 'Pricing']
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first'), self.categorical_features)
            ],
            remainder='passthrough'  # Keep numerical columns as they are
        )

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Food to Students Ratio'] = df['Quantity of Food'] / df['Number of Students']
        df['Is Weekend'] = df['Day of the Week'].apply(lambda x: 1 if x in ['Sunday', 'Saturday'] else 0)
        return df

    def fit_transform(self, X: pd.DataFrame):
        return self.preprocessor.fit_transform(X)

    def transform(self, X: pd.DataFrame):
        return self.preprocessor.transform(X)

# 3. Model Training Class
class ModelTrainer:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor

        # Base models for stacking
        self.base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse'))
        ]

        # Final stacking model
        self.stacking_model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse')
        )

        # Create a pipeline that includes preprocessing and the model
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor.preprocessor),
            ('model', self.stacking_model)
        ])

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        return X_test, y_test  # Return test data for evaluation

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        predictions = self.pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

# 4. Prediction Pipeline Class
class PredictionPipeline:
    def __init__(self, model_trainer: ModelTrainer):
        self.pipeline = model_trainer.pipeline

    def predict(self, input_data: Dict):
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])
        input_df['Food to Students Ratio'] = input_df['quantity_of_food'] / input_df['number_of_students']
        input_df['Is Weekend'] = input_df['day_of_week'].apply(lambda x: 1 if x in ['Sunday', 'Saturday'] else 0)

        # Make predictions
        prediction = self.pipeline.predict(input_df)

        # Return rounded prediction
        return float(np.round(prediction[0], 2))

# 5. Main Train and Prediction Pipeline
def main_train_pipeline():
    # Load dataset
    df = pd.read_csv('balanced_food_wastage_data.csv')

    # Separate features and target variable
    X = df.drop(columns=['Wastage Food Amount'])
    y = df['Wastage Food Amount']

    # Create a preprocessor
    preprocessor = DataPreprocessor()

    # Apply feature engineering
    X = preprocessor.feature_engineering(X)

    # Train model
    model_trainer = ModelTrainer(preprocessor)
    X_test, y_test = model_trainer.train(X, y)

    # Evaluate model
    mse, r2 = model_trainer.evaluate(X_test, y_test)
    print(f'MSE: {mse}, R^2: {r2}')

def main_prediction_pipeline():
    # Mock input data for prediction
    input_data = {
        "quantity_of_food": 100,
        "number_of_students": 50,
        "type_of_food": "Rice",
        "day_of_week": "Monday",
        "storage_conditions": "Refrigerated",
        "seasonality": "Winter",
        "pricing": 200
    }

    # Use trained model for prediction
    model_trainer = ModelTrainer(DataPreprocessor())  # Assuming pre-trained model
    prediction_pipeline = PredictionPipeline(model_trainer)
    prediction = prediction_pipeline.predict(input_data)
    print(f'Predicted Wastage: {prediction}')

if __name__ == "__main__":
    main_train_pipeline()
    main_prediction_pipeline()
