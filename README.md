# Stock Price Prediction Using Machine Learning

This project leverages machine learning algorithms to predict stock prices based on historical data, aiming to assist investors in minimizing investment risks. The best-performing algorithm is selected through model evaluation, and a user-friendly interactive webpage is developed for users to predict stock prices with ease.

> **Note**: This project was completed in 2022, and the results are based on data and models available at that time. Predictions and model accuracy may vary if the project is rerun with updated data or model improvements available in recent years.

**Project Documentation**: For detailed information about this project, kindly refer to the [Project_Overview](https://github.com/SHETTY-DHIRAJ/Stock-Price-Prediction-Using-Machine-Learning/blob/main/Dependent%20Resources/Project_Overview.pdf) document.

## Project Overview

- **Objective**: Predict stock prices for the next 10 days using historical stock market data.
- **Algorithms**: Linear Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, and Long Short-Term Memory (LSTM).
- **Best Performing Model**: LSTM was identified as the best algorithm and is used in the interactive webpage for predictions.
- **Technologies**: Python, Streamlit for web development, various machine learning libraries for algorithm implementation.

## Screenshots

### 1. Research Related Screenshots

<p align="center">
  <img src="https://github.com/SHETTY-DHIRAJ/Stock-Price-Prediction-Using-Machine-Learning/blob/main/Dependent%20Resources/The%20R-Square%20Error%2C%20MAE%2C%20RMS%20of%20all%20five%20ML%20algorithms%20in%20use.jpg" alt="The R-Square Error, MAE, RMS of all five ML algorithms in use" width="800">
</p>
<p align="center"><em>The R-Square Error, MAE, RMS of all five ML algorithms in use</em></p>

<p align="center">----</p>

<p align="center">
  <img src="https://github.com/SHETTY-DHIRAJ/Stock-Price-Prediction-Using-Machine-Learning/blob/main/Dependent%20Resources/LSTM%20(Long%20and%20Short%20Memory)%20Algorithm%20Predictions.jpg" alt="LSTM (Long and Short Memory) Algorithm Predictions" width="800">
</p>
<p align="center"><em>LSTM (Long and Short Memory) Algorithm Predictions</em></p>

<p align="center">----</p>

<p align="center">
  <img src="https://github.com/SHETTY-DHIRAJ/Stock-Price-Prediction-Using-Machine-Learning/blob/main/Dependent%20Resources/Next%20ten%20days%20Stock%20Price%20prediction%20of%20all%20five%20ML%20algorithms%20in%20use.jpg" alt="Next ten days Stock Price prediction of all five ML algorithms in use" width="600">
</p>
<p align="center"><em>Next ten days Stock Price prediction of all five ML algorithms in use</em></p>

### 2. Streamlet Webpage Related Screenshots
<p align="center">
  <img src="https://github.com/SHETTY-DHIRAJ/Stock-Price-Prediction-Using-Machine-Learning/blob/main/Dependent%20Resources/The%20Start%20Page%20of%20Streamlit%20Webpage.jpg" alt="The Start Page of Streamlit Webpage" width="800">
</p>
<p align="center"><em>The Start Page of Streamlit Webpage</em></p>

<p align="center">----</p>

<p align="center">
  <img src="https://github.com/SHETTY-DHIRAJ/Stock-Price-Prediction-Using-Machine-Learning/blob/main/Dependent%20Resources/The%20Chart%20Containing%20the%20Actual%20and%20Predicted%20Price%20of%20the%20Training%20and%20Testing%20dataset%20in%20Webpage.jpg" alt="The Chart Containing the Actual and Predicted Price of the Training and Testing dataset in Webpage" width="800">
</p>
<p align="center"><em>The Chart Containing the Actual and Predicted Price of the Training and Testing dataset in Webpage</em></p>

<p align="center">----</p>

<p align="center">
  <img src="https://github.com/SHETTY-DHIRAJ/Stock-Price-Prediction-Using-Machine-Learning/blob/main/Dependent%20Resources/Next%20ten%20days%20Stock%20Price%20prediction%20of%20LSTM%20in%20Webpage.jpg" alt="Next ten days Stock Price prediction of LSTM in Webpage" width="500">
</p>
<p align="center"><em>Next ten days Stock Price prediction of LSTM in Webpage</em></p>

## Features

1. **Data Collection and Preprocessing**
   - Stock data is collected from Yahoo Finance and Nasdaq.
   - Preprocessing steps include data cleaning, handling missing values, and feature selection.

2. **Algorithm Implementation**
   - Multiple algorithms are applied to the dataset, and their performance is evaluated based on metrics like R-Square (R2 Score), Mean Absolute Error (MAE), and Root Mean Square Error (RMSE).
   - LSTM outperformed other algorithms and is used in the prediction model.

3. **Interactive Webpage**
   - A Streamlit-based webpage allows users to input a stock ticker.
   - The webpage provides historical data and a 10-day price prediction for the selected stock.

## Project Structure

- **Notebooks**: Jupyter notebook containing overall research code, located in the `research.ipynb` file.

- **Webpage**: A Streamlit application for user interaction, located in the `app.py` file.
  - **Run the Streamlit application:**
     ```bash
     streamlit run app.py

## Prerequisites
   - Python 3.8 or higher
   - Jupyter Notebook for model training and evaluation
   - Streamlit for building the interactive webpage
   - Libraries: pandas, matplotlib, scikit-learn, Keras (for LSTM implementation), Streamlit

## System Architecture
The system consists of four main phases:

- **Data Splitting**: Data is preprocessed, cleaned, and split into training and testing sets.
- **Model Training**: Algorithms like Linear Regression, KNN, SVM, Random Forest, and LSTM are trained on historical stock data.
- **Model Evaluation**: Models are evaluated using metrics like R2 Score, MAE, and RMSE.
- **Interactive Webpage**: The best-performing model (LSTM) is utilized on a Streamlit webpage for user interaction.

## Evaluation Metrics
The following metrics were used to evaluate the algorithms:

- **R-Square (R2 Score)**: Measures how well the model fits the data.
- **Mean Absolute Error (MAE)**: Average of absolute differences between predicted and actual values.
- **Root Mean Square Error (RMSE)**: Square root of the average of squared differences between predicted and actual values.

## Results

- LSTM showed the highest accuracy among the tested algorithms.
- The interactive webpage uses LSTM to provide stock price predictions, allowing users to get historical and future stock price data by simply entering the stock ticker.

## Future Enhancements

- Integrate sentiment analysis of financial news for enhanced prediction.
- Add functionality to consider external factors like economic indicators.
- Enable users to view predictions for multiple stocks simultaneously.

## Acknowledgements

This project is inspired by machine learning techniques and applications in stock market prediction. Data was sourced from [Yahoo Finance](https://finance.yahoo.com/) and [Nasdaq](https://www.nasdaq.com/).
