# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load your trained model (make sure you have the correct path to the model)
# model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract form data from the POST request
#     data = request.form

#     # ---- Capture all required input features from the form ----
#     type_of_food = data.get('type_of_food')  # Correctly match form field name
#     day_of_week = data.get('day_of_week')
#     storage_conditions = data.get('storage_conditions')
#     seasonality = data.get('seasonality')
#     pricing = data.get('pricing')

#     print("Type of food : ", type_of_food)

#     # Ensure that all the form fields are filled, else return an error
#     if not type_of_food or not day_of_week or not storage_conditions or not seasonality or not pricing:
#         return render_template('predict.html', prediction="Please fill in all the fields.")

#     # ---- Handle cases where numeric fields may be missing ----
#     try:
#         quantity_of_food = float(data.get('quantity_of_food', 0))  # Correctly match form field name
#         number_of_students = int(data.get('number_of_students', 0))  # Correctly match form field name
#     except ValueError:
#         return render_template('predict.html', prediction="Please provide valid numbers for Quantity of Food and Number of Students.")

#     # ---- Map categorical pricing to numeric values ----
#     pricing_map = {'Low': 1, 'Medium': 2, 'High': 3}
#     pricing_numeric = pricing_map.get(pricing, 0)  # Default to 0 if 'pricing' is not in the map

#     # ---- Additional calculated features ----
#     # Calculate the Food to Students Ratio (as an additional feature)
#     food_to_students_ratio = quantity_of_food / number_of_students if number_of_students > 0 else 0
    
#     # Identify if the day is a weekend (0 for weekday, 1 for weekend)
#     is_weekend = 1 if day_of_week in ['Sunday', 'Saturday'] else 0
    
#     # ---- Create an array of all 9 features ----
#     input_features = [
#         type_of_food, 
#         day_of_week, 
#         storage_conditions, 
#         seasonality, 
#         pricing_numeric,  # Use the mapped numeric value
#         quantity_of_food, 
#         number_of_students, 
#         food_to_students_ratio, 
#         is_weekend
#     ]

#     # Convert input features to numpy array and reshape it for prediction
#     input_features = np.array(input_features).reshape(1, -1)

#     # ---- Predict using the model ----
#     try:
#         prediction = model.predict(input_features)  # Ensure your model accepts this input
#         prediction_result = prediction[0]  # Get the first element for the result
#     except ValueError as e:
#         print("Error during prediction:", e)
#         return render_template('predict.html', prediction="Error in input data.")

#     # Return the prediction result to the template
#     return render_template('predict.html', prediction=f"The predicted food waste is: {prediction_result:.2f} kg.")

# if __name__ == "__main__":
#     app.run(debug=True)






# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from modularized_model import DataPreprocessor, ModelTrainer, PredictionPipeline

# app = Flask(__name__)

# # Initialize preprocessor and model trainer
# preprocessor = DataPreprocessor()
# model_trainer = ModelTrainer(preprocessor)

# # Train the model pipeline (this assumes you have a dataset for training)
# # Ensure your dataset path is correct
# df = pd.read_csv('balanced_food_wastage_data.csv')
# X = df.drop(columns=['Wastage Food Amount'])
# y = df['Wastage Food Amount']
# X = preprocessor.feature_engineering(X)
# model_trainer.train(X, y)

# # Initialize the prediction pipeline
# prediction_pipeline = PredictionPipeline(model_trainer)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract form data from the POST request
#     data = request.form

#     # Capture all required input features from the form
#     input_data = {
#         "type_of_food": data.get('type_of_food'),
#         "day_of_week": data.get('day_of_week'),
#         "storage_conditions": data.get('storage_conditions'),
#         "seasonality": data.get('seasonality'),
#         "pricing": float(data.get('pricing')),  # Assume pricing is numerical
#         "quantity_of_food": float(data.get('quantity_of_food', 0)),
#         "number_of_students": int(data.get('number_of_students', 0))
#     }

#     # Ensure that all the form fields are filled, else return an error
#     if not all(input_data.values()):
#         return render_template('predict.html', prediction="Please fill in all the fields.")

#     # ---- Predict using the modularized prediction pipeline ----
#     try:
#         prediction = prediction_pipeline.predict(input_data)
#         prediction_result = prediction  # The result from the prediction pipeline
#     except Exception as e:
#         print("Error during prediction:", e)
#         return render_template('predict.html', prediction="Error in input data.")

#     # Return the prediction result to the template
#     return render_template('predict.html', prediction=f"The predicted food waste is: {prediction_result:.2f} kg.")

# if __name__ == "__main__":
#     app.run(debug=True)




from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the React frontend

# Load your machine learning model
model  = pickle.load(open("model.pkl", "rb"))

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data sent from the frontend
    try:
        # Extract and prepare data for prediction
        type_of_food = data['typeOfFood']
        number_of_students = int(data['numberOfStudents'])
        day_of_week = data['dayOfWeek']
        quantity_of_food = float(data['quantityOfFood'])
        storage_conditions = data['storageConditions']
        seasonality = data['seasonality']
        pricing = data['pricing']

        # Convert categorical data to numeric values or use one-hot encoding if needed
        # Example below assumes simple numerical encoding for demonstration purposes

        # input_features = [
        #     convert_type_of_food(type_of_food),
        #     number_of_students,
        #     convert_day_of_week(day_of_week),
        #     quantity_of_food,
        #     convert_storage_conditions(storage_conditions),
        #     convert_seasonality(seasonality),
        #     convert_pricing(pricing)
        # ]

        # input_features = np.array(input_features).reshape(1, -1)

        # print("--------------->>>>>>>>> : ", type_of_food)

             # Ensure all data is present
        if not all([type_of_food, number_of_students, day_of_week, quantity_of_food, storage_conditions, seasonality, pricing]):
            return jsonify({'error': 'All fields are required'}), 400
    
        # Making new required features
        Food_to_Students_Ratio = int(quantity_of_food) / int(number_of_students)
        Is_Weekend = False
        if day_of_week == "Saturday" or day_of_week == "Sunday":
            Is_Weekend = 1
        else :
            Is_Weekend = 0


        # Prepare data in a format suitable for the pipeline
        input_data = pd.DataFrame({
            'Type of Food': [type_of_food],
            'Number of Students': [int(number_of_students)],
            'Day of the Week': [day_of_week],
            'Quantity of Food': [float(quantity_of_food)],
            'Storage Conditions': [storage_conditions],
            'Seasonality': [seasonality],
            'Pricing': [pricing],
            'Food to Students Ratio' : [float(Food_to_Students_Ratio)],
            'Is Weekend': Is_Weekend 
        })

        # Predict with the model
        prediction = model.predict(input_data)
        predicted_waste = round(prediction[0], 2)

        # print("--------->>>>>>>>>", prediction[0])

        # Return prediction to frontend
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        # Handle errors
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

# Helper functions for data conversion (you may need to adjust these based on your data)
def convert_type_of_food(food_type):
    # Example conversion, replace with actual encoding logic
    food_mapping = {'Fruits': 0, 'Vegetables': 1, 'Baked Goods': 2, 'Dairy Products': 3}
    return food_mapping.get(food_type, -1)

def convert_day_of_week(day):
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    return day_mapping.get(day, -1)

def convert_storage_conditions(condition):
    condition_mapping = {'Refrigerated': 0, 'Room Temperature': 1}
    return condition_mapping.get(condition, -1)

def convert_seasonality(season):
    season_mapping = {'Summer': 0, 'Winter': 1, 'All Seasons': 2}
    return season_mapping.get(season, -1)

def convert_pricing(price):
    price_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    return price_mapping.get(price, -1)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
