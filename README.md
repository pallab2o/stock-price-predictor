<div align="center">
  <img src="logo.png" alt="Logo" width="200" height="200">
</div>

# Stock Predictor

This is a web application built with Streamlit that allows users to predict stock prices using historical data. Users can input a stock symbol, adjust advanced settings, and visualize the predictions along with the actual stock value over time.

## Features

- **Prediction Model:** The app utilizes a Random Forest Classifier model to predict stock prices based on historical data.
- **Profit Calculation:** It includes a profit calculation function that estimates the profit or loss based on the predictions.
- **Advanced Settings:** Users can adjust advanced settings such as the start index and step size for the prediction range to customize their analysis.
- **Visualization:** The app provides interactive plots to visualize the predictions, actual stock values, and other relevant metrics.

## How It Works

1. **Input Stock Symbol:** Users enter the stock symbol (e.g., AAPL for Apple) in the text box on the left sidebar.
2. **Adjust Settings:** Users can adjust advanced settings such as the start index and step size for the prediction range.
3. **Run Prediction:** After entering the stock symbol and adjusting settings, users click the "Run" button to initiate the prediction process.
4. **View Results:** The app displays the predictions, actual stock values, and other relevant metrics such as model accuracy and profit/loss.

## Prediction Model

The app utilizes a Random Forest Classifier model trained on historical stock price data obtained from Yahoo Finance (via the `yfinance` library). The model predicts whether the stock price will increase or decrease based on features such as closing price, high, low, open, and volume.

## Profit Calculation

The profit calculation function estimates the profit or loss based on the predictions generated by the model. It compares the predicted stock price movement with the actual stock price movement and calculates the difference between the closing and opening prices.

## Getting Started

To run the app locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the Streamlit app by executing `streamlit run app.py` in your terminal.
4. Access the app in your web browser at `http://localhost:8501`.

## Deployed Website

The app is already deployed and accessible online. You can try it out at [Stock Price Prediction App](https://stock-predictorv2.streamlit.app/#stock-predictor).

## Feedback and Contributions

Feedback, bug reports, and contributions are welcome! If you encounter any issues or have suggestions for improvement, please [open an issue](https://github.com/yourusername/stock-price-prediction-app/issues) or [submit a pull request](https://github.com/yourusername/stock-price-prediction-app/pulls).

## Acknowledgments

This project was inspired by the need for a simple and user-friendly tool to predict stock prices and visualize the results. Special thanks to the Streamlit and Yahoo Finance communities for their contributions and support.
