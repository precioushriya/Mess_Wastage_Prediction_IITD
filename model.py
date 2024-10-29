# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from xgboost import XGBRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# import joblib

# # Load the dataset
# df = pd.read_csv('balanced_food_wastage_data.csv')  # Adjust the path as necessary

# # Feature Engineering
# df['Food to Students Ratio'] = df['Quantity of Food'] / df['Number of Students']
# df['Is Weekend'] = df['Day of the Week'].apply(lambda x: 1 if x in ['Sun', 'Sat'] else 0)

# # Preparing features and target
# X = df.drop(columns=['Wastage Food Amount'])
# y = df['Wastage Food Amount']

# # Define preprocessing for categorical features
# categorical_features = ['Type of Food', 'Day of the Week', 'Storage Conditions', 'Seasonality', 'Pricing']
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(drop='first'), categorical_features)
#     ],
#     remainder='passthrough'
# )

# # Base models
# base_models = [
#     ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
#     ('xgb', XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse'))
# ]

# # Stacking regressor
# stacking_regressor = StackingRegressor(
#     estimators=base_models,
#     final_estimator=XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse')
# )

# # Pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', stacking_regressor)
# ])

# # Train the model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# pipeline.fit(X_train, y_train)

# # Save the model
# joblib.dump(pipeline, 'model.pkl')
# print("Model saved as model.pkl")




import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

class PredictionPipeline:
    def __init__(self, csv_path):
        # Load dataset during initialization
        self.df = pd.read_csv(csv_path)
        self.pipeline = None  # To store the trained pipeline

        # Preprocess dataset (Feature Engineering)
        self.df['Food to Students Ratio'] = self.df['Quantity of Food'] / self.df['Number of Students']
        self.df['Is Weekend'] = self.df['Day of the Week'].apply(lambda x: 1 if x in ['Sun', 'Sat'] else 0)

        # Prepare features and target variable
        self.X = self.df.drop(columns=['Wastage Food Amount'])
        self.y = self.df['Wastage Food Amount']

    def create_pipeline(self):
        # Define preprocessing for categorical features
        categorical_features = ['Type of Food', 'Day of the Week', 'Storage Conditions', 'Seasonality', 'Pricing']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ],
            remainder='passthrough'  # Keeps numerical columns unchanged
        )

        # Define base models
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse'))
        ]

        # Create stacking regressor
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse')
        )

        # Create pipeline with preprocessing and stacking regressor
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', stacking_regressor)
        ])

    def train_model(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Train the model on the training data
        self.pipeline.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = self.pipeline.predict(X_test)

        # Evaluate the model using Mean Squared Error and R2 Score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Successfully Model trained. MSE: {mse:.2f}, R2 Score: {r2:.2f}, Accuracy: {r2 * 100:.2f}%")

        with open("model.pkl", "wb") as model_file:
            pickle.dump(self.pipeline, model_file)


    def predict(self, input_data):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        prediction = self.pipeline.predict(input_df)
        return prediction[0]  # Return the prediction value


if __name__ == "__main__" :
    path_to_input = "balanced_food_wastage_data.csv"
    pipeline = PredictionPipeline(path_to_input)
    pipeline.create_pipeline()
    pipeline.train_model()






