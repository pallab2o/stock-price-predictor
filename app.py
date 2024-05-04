import streamlit as st
import yfinance as yf
import pandas as pd

import yfinance as yf

def model(symbol):                                                   
    trid = yf.Ticker(symbol)
    trid_hist = trid.history(period="max")

    data = trid_hist[["Close"]]
    data = data.rename(columns = {'Close': 'Actual_Close'})
    data["Target"]= trid_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

    trid_prev = trid_hist.copy()

    trid_prev = trid_prev.shift(1)

    trid_prev.head(5)

    predictors = ["Close", "High", "Low", "Open", "Volume"]
    data = data.join(trid_prev[predictors]).iloc[1:]

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, min_samples_split=200)

    import pandas as pd

    start = 1000
    step = 750
    def backtest(data, model, predictors, start, step):
        predictions = []
        for i in range(start, data.shape[0], step):
        
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            
            model.fit(train[predictors], train["Target"])
            
            preds = model.predict_proba(test[predictors])[:,1]
            preds = pd.Series(preds, index = test.index)
            preds[preds > .6] = 1
            preds[preds <= .6] = 0
            
            combined = pd.concat({"Target": test["Target"], "Predictions":preds}, axis=1)
        
            predictions.append(combined)
        
        predictions = pd.concat(predictions)
        return data, predictions

    weekly_mean = data.rolling(7).mean()
    quarterly_mean = data.rolling(90).mean()
    annual_mean = data.rolling(365).mean()

    weekly_trend = data.shift(1).rolling(7).mean()["Target"]

    data["weekly_mean"] = weekly_mean["Close"]/data["Close"]
    data["quarterly_mean"] = quarterly_mean["Close"]/data["Close"]
    data["annual_mean"] = annual_mean["Close"]/data["Close"]

    data["annual_weekly_mean"] = data["annual_mean"]/data["weekly_mean"]
    data["annual_quarterly_mean"] = data["annual_mean"]/data["quarterly_mean"]
    data["quarterly_weekly_mean"] = data["quarterly_mean"]/data["weekly_mean"]

    full_predictors = predictors + ["weekly_mean","quarterly_mean","annual_mean","annual_weekly_mean","annual_quarterly_mean","quarterly_weekly_mean"]

    return backtest(data, model, full_predictors, start, step)

def profitcalc(predictions, data):
    profit = 0
    data = data.iloc[100:]
    # Adjust data index to match predictions index
    for i in range(0, predictions.shape[0]):
        if predictions.iloc[i]["Predictions"] == 1:
            # Get the corresponding index from predictions
            prediction_index = predictions.index[i]
            # Use the prediction index to retrieve data
            data_index = data.index[data.index.get_loc(prediction_index)]
            profit += data.loc[data_index, 'Close'] - data.loc[data_index, 'Open']
    return profit

def main():
    st.title('Stock Price Prediction App')

    # User input for stock symbol
    symbol = st.text_input('Enter Stock Symbol (e.g., AAPL, MSFT)', 'AAPL')
    # Sidebar for advanced settings
    with st.sidebar:
        st.subheader('Advanced Settings')
        start = st.number_input('Start', value=1000)
        step = st.number_input('Step', value=750)
    
    if symbol:
        # Attempt to retrieve data for the provided stock symbol
        try:
            data, predictions = model(symbol)
            profit = profitcalc(predictions, data)

            from sklearn.metrics import precision_score 
            precision = 100 * precision_score(predictions["Target"], predictions["Predictions"])

            # Display predictions
            st.sidebar.subheader('Predictions DataFrame')
            st.sidebar.write(predictions)

            # Plot the predictions vs target and the actual stock value over time
            
            import matplotlib.pyplot as plt

             # Plot the graph
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(predictions.index, predictions['Target'], label='Target', marker='o')
            ax.plot(predictions.index, predictions['Predictions'], label='Predictions', marker='x')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.set_title('Target vs Predictions')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Plot the data line
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            data["Close"].plot.line(ax=ax2, use_index=True)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Close Value')
            ax2.set_title('Close Value Over Time')
            ax2.grid(True)

            # Display the plots using Streamlit
            st.pyplot(fig2)
            st.pyplot(fig)
            

            # Display the profit value centered below both
            st.markdown('<h2 style="text-align: center; color: blue;">Model Accuracy: {:.2f}%</h2>'.format(precision), unsafe_allow_html=True)
            if profit > 0:
                st.markdown('<h2 style="text-align: center; color: green;">Profit: ${:.2f}</h2>'.format(profit), unsafe_allow_html=True)
            else:
                st.markdown('<h2 style="text-align: center; color: red;">Loss: ${:.2f}</h2>'.format(abs(profit)), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()