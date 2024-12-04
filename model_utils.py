import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


def build_and_train_model(X_train, y_train, X_val, y_val, input_shape):
    """
    Build, compile, and train an LSTM-based model.

    Parameters:
        X_train (np.array): Training input data.
        y_train (np.array): Training target data.
        X_val (np.array): Validation input data.
        y_val (np.array): Validation target data.
        input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
        model (tensorflow.keras.Model): Trained Keras model.
        history (tensorflow.keras.callbacks.History): Training history.
    """
    model = Sequential([
        Input(shape=input_shape),  # Input layer with specified shape
        LSTM(units=100, return_sequences=True),  # First LSTM layer with 100 units
        Dropout(0.2),  # Dropout for regularization
        LSTM(units=50),  # Second LSTM layer with 50 units
        Dropout(0.2),  # Dropout for regularization
        Dense(1)  # Output layer predicting a single value
    ])

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping to prevent overfitting and save the best weights
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, start_from_epoch=50)

    # Train the model on the training data and validate on validation data
    history = model.fit(
        X_train, y_train,
        epochs=100,  # Train for a maximum of 100 epochs
        batch_size=32,  # Use batches of 32 samples
        validation_data=(X_val, y_val),  # Validation data for monitoring
        callbacks=[early_stopping]  # Include early stopping callback
    )

    return model, history


def random_walk_baseline(y_test):
    """
    Baseline: Predict next price using the current price plus random noise.

    Parameters:
        y_test (np.array): Actual target values.

    Returns:
        random_walk_pred (np.array): Predicted values using the random walk baseline.
        random_walk_actual (np.array): Actual target values for comparison.
    """
    np.random.seed(42)  # Ensure reproducibility

    # Calculate daily changes (deltas)
    deltas = np.diff(y_test)
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)

    # Generate random noise based on historical deltas
    noise = np.random.normal(mean_delta, std_delta, size=len(y_test) - 1)

    # Predict using the random walk formula
    random_walk_pred = y_test[:-1] + 2 * noise

    return random_walk_pred, y_test[1:]


def evaluate_model(model, X_test, y_test, scaler, scaled_df, look_back, ticker, output_dir="evaluation_data"):
    """
    Evaluate the model and generate visualizations for predictions and errors.

    Parameters:
        model (tensorflow.keras.Model): Trained Keras model.
        X_test (np.array): Test input data.
        y_test (np.array): Test target data.
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler for inverse transforming the data.
        scaled_df (pd.DataFrame): Scaled DataFrame of features.
        look_back (int): Number of previous time steps used in the model.
        ticker (str): Stock ticker symbol.
        output_dir (str): Directory to save evaluation outputs.

    Returns:
        None
    """
    output_dir += f"/{ticker}"

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Inverse transform the predictions and actual values
    num_features = scaled_df.shape[1]  # Number of features in the scaled data
    y_test_actual = scaler.inverse_transform(
np.concatenate((np.zeros((len(y_test), num_features - 1)), y_test.reshape(-1, 1)), axis=1)
    )[:, -1]  # Extract actual values
    y_pred_actual = scaler.inverse_transform(
        np.concatenate((np.zeros((len(y_pred), num_features - 1)), y_pred), axis=1)
    )[:, -1]  # Extract predicted values

    # Calculate random walk baseline metrics
    random_walk_pred, random_walk_actual = random_walk_baseline(y_test_actual)
    random_walk_rmse = np.sqrt(mean_squared_error(random_walk_actual, random_walk_pred))
    random_walk_mape = mean_absolute_percentage_error(random_walk_actual, random_walk_pred)
    random_walk_r_squared = r2_score(random_walk_actual, random_walk_pred)

    # Calculate evaluation metrics for the model
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
    r_squared = r2_score(y_test_actual, y_pred_actual)

    # Save metrics to a text file
    file_path = output_dir + "/metrics.txt"
    with open(file_path, "w") as file:
        file.write(f"--- {ticker} ---\n")
        file.write(f"RMSE: {rmse:.2f}\n")
        file.write(f"Baseline RMSE: {random_walk_rmse:.2f}\n")
        file.write(f"MAPE: {mape:.2f}\n")
        file.write(f"Baseline MAPE: {random_walk_mape:.2f}\n")
        file.write(f"R-squared: {r_squared:.4f}\n")
        file.write(f"Baseline R-squared: {random_walk_r_squared:.4f}")

    # Plot residuals (differences between actual and predicted values)
    residuals = y_test_actual - y_pred_actual
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, alpha=0.7, color='blue', label='Residuals')
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.title(f"{ticker} Residual Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.legend()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"residuals.png"))
    plt.close()

    # Scatter plot of predictions vs actuals
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.7, color='blue', label='Predictions')
    plt.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], color='red',
             linestyle='--', label='Ideal Fit')
    plt.title(f"{ticker} Predicted vs Actual")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"predicted_vs_actual.png"))
    plt.close()

    # Visualize the predictions alongside actual values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Price', color='blue')
    plt.plot(y_pred_actual, label='Predicted Price', color='red')
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    output_path = os.path.join(output_dir, f"prediction.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

    # Compare model predictions with random walk baseline
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Price', color='blue')
    plt.plot(y_pred_actual, label='Model Prediction', color='red')
    plt.plot(np.append([np.nan], random_walk_pred), label='Random Walk Baseline', color='green')
    plt.title(f"{ticker} Stock Price Prediction with Random Walk Baseline")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"random_walk_comparison.png"))
    plt.close()
