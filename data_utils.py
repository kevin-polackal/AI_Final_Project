import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker and merge it with S&P 500 data.

    Parameters:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data retrieval (format: 'YYYY-MM-DD').
        end_date (str): End date for data retrieval (format: 'YYYY-MM-DD').

    Returns:
        pd.DataFrame: Merged DataFrame containing stock data and S&P 500 data.
    """
    # Fetch stock data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Fetch S&P 500 index data as an external factor
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)[['Close']]
    sp500_data.rename(columns={'Close': 'SP500_Close'}, inplace=True)  # Rename column for clarity

    # Merge stock data with S&P 500 data on their indices (dates)
    df = stock_data.merge(sp500_data, left_index=True, right_index=True, how='left')

    # Handle missing values by dropping rows with NaNs
    df.dropna(inplace=True)

    return df


def preprocess_data(df, features, look_back):
    """
    Preprocess stock data to create time series sequences for model input.

    Parameters:
        df (pd.DataFrame): DataFrame containing the stock and external factor data.
        features (list): List of feature column names to use for modeling.
        look_back (int): Number of previous time steps to include in each sequence.

    Returns:
        tuple: (X, y, scaler, scaled_df)
            X (np.array): Array of input sequences.
            y (np.array): Array of target values.
            scaler (MinMaxScaler): Fitted scaler for data normalization.
            scaled_df (pd.DataFrame): Scaled version of the input DataFrame.
    """
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)

    X, y = [], []
    for i in range(look_back, len(scaled_df)):
        # Append the sequence of features (look_back time steps)
        X.append(scaled_df.iloc[i - look_back:i].values)
        # Append the target value (closing price at the current time step)
        y.append(scaled_df.iloc[i]['Close'])

    # Convert lists to NumPy arrays for compatibility with machine learning models
    X = np.array(X)
    y = np.array(y)

    return X, y, scaler, scaled_df
