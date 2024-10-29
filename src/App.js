import React, {useEffect, useState} from 'react';
import { motion } from 'framer-motion';
import { FaUtensils, FaUsers, FaCalendarAlt, FaWeight, FaTemperatureHigh, FaSun, FaDollarSign } from 'react-icons/fa';
import Lottie from 'lottie-react';
// import wasteBackground from './img2.jpg'; // Original background image
import animationData from './wasteAnimation.json'; // Import your Lottie animation file
import './App.css';

const App = () => {
  const [formData, setFormData] = useState({
    typeOfFood: '',
    numberOfStudents: '',
    dayOfWeek: '',
    quantityOfFood: '',
    storageConditions: '',
    seasonality: '',
    pricing: '',
  });

  const [prediction, setPrediction] = useState(null); // State to store prediction result

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const result = await response.json();
      console.log('Prediction result:', result);
      setPrediction(result.prediction); // Store prediction result in state
    } catch (error) {
      console.error('Error fetching prediction:', error);
      alert("Error fetching prediction. Please try again.");
    }
  };
  

  return ( 
    <div className="app-container">
      {/* Original Background */}
      <div className="background-image" />
      
      {/* Lottie Animation */}
      <div className="lottie-background">
        <Lottie animationData={animationData} loop={true} />
      </div>

      <div className="content">
        <motion.h1
          className="header"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Waste Prediction Dashboard
        </motion.h1>
        <motion.p
          className="intro"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          Welcome to the Waste Calculator for IIT Delhi hostels! Help us track and minimize waste
          by providing information about daily food consumption. Let’s work together to make our hostel a greener place.
        </motion.p>

        <motion.form
          className="form"
          onSubmit={handleSubmit}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          {/* Form Fields */}
          <div className="form-card">
            <FaUtensils className="icon" />
            <label htmlFor="typeOfFood">Type of Food</label>
            <select name="typeOfFood" onChange={handleChange}>
              <option value="">Select Type</option>
              <option value="Fruits">Fruits</option>
              <option value="Vegetables">Vegetables</option>
              <option value="Baked Goods">Baked Goods</option>
              <option value="Dairy Products">Dairy Products</option>
            </select>
          </div>

          <div className="form-card">
            <FaUsers className="icon" />
            <label htmlFor="numberOfStudents">Number of Students</label>
            <input type="number" name="numberOfStudents" onChange={handleChange}
              placeholder="Enter number of students"
            />
          </div>

          <div className="form-card">
            <FaCalendarAlt className="icon" />
            <label htmlFor="dayOfWeek">Day of the Week</label>
            <select name="dayOfWeek" onChange={handleChange}>
              <option value="">Select Day</option>
              <option value="Mon">Monday</option>
              <option value="Tue">Tuesday</option>
              <option value="Wed">Wednesday</option>
              <option value="Thurs">Thursday</option>
              <option value="Fri">Friday</option>
              <option value="Sat">Saturday</option>
              <option value="Sun">Sunday</option>
            </select>
          </div>

          <div className="form-card">
            <FaWeight className="icon" />
            <label htmlFor="quantityOfFood">Quantity of Food (in kg)</label>
            <input
              type="number"
              name="quantityOfFood"
              onChange={handleChange}
              placeholder="Enter quantity of food"
            />
          </div>

          <div className="form-card">
            <FaTemperatureHigh className="icon" />
            <label htmlFor="storageConditions">Storage Conditions</label>
            <select name="storageConditions" onChange={handleChange}>
              <option value="">Select Condition</option>
              <option value="Refrigerated">Refrigerated</option>
              <option value="Room Temperature">Room Temperature</option>
            </select>
          </div>

          <div className="form-card">
            <FaSun className="icon" />
            <label htmlFor="seasonality">Seasonality</label>
            <select name="seasonality" onChange={handleChange}>
              <option value="">Select Season</option>
              <option value="Summer">Summer</option>
              <option value="Winter">Winter</option>
              <option value="All Seasons">All Seasons</option>
            </select>
          </div>

          <div className="form-card">
            <FaDollarSign className="icon" />
            <label htmlFor="pricing">Pricing</label>
            <select name="pricing" onChange={handleChange}>
              <option value="">Select Pricing</option>
              <option value="Low">Low</option>
              <option value="Moderate">Moderate</option>
              <option value="High">High</option>
            </select>
          </div>

          <button type="submit" className="submit-btn">
            Submit
          </button>
        </motion.form>

        {prediction && (
          <motion.div
            className="prediction-result"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <h2>Predicted Waste Amount: {prediction} kg</h2>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default App;
