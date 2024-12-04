import subprocess
import sys
from data_utils import fetch_data, preprocess_data
from model_utils import build_and_train_model, evaluate_model
import matplotlib.pyplot as plt
import os


def install_dependencies(dependencies_file):
    """Install dependencies listed in a text file."""
    try:
        with open(dependencies_file, 'r') as file:
            dependencies = file.read().splitlines()

        for dependency in dependencies:
            if dependency.strip():  # Skip empty lines
                print(f"Installing {dependency}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
    except FileNotFoundError:
        print(f"Error: {dependencies_file} not found.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)


def plot_training_history(history, output_dir="evaluation_data", ticker="default"):
    """
       Plot and save the training loss curve.

       Parameters:
           history (tensorflow.keras.callbacks.History): Training history object.
           output_dir (str): Directory where the plot will be saved.
           ticker (str): Stock ticker name for labeling the plot.

       Returns:
           None
    """
    output_dir += f"/{ticker}"

    # Extract training loss from history
    train_loss = history.history['loss']
    epochs = range(1, len(train_loss) + 1)

    # Create a plot for training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', alpha=0.7)
    plt.title(f'{ticker} Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"loss.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def gather_predictions():
    """Fetch data, preprocess it, train models, and evaluate predictions."""
    # Define parameters for the prediction task
    tickers = ['AAPL', 'MSFT', 'AMZN', 'JNJ', 'JPM']  # Stock tickers to process
    start_date = '2013-01-01'  # Start date for historical data
    end_date = '2023-10-01'  # End date for historical data
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SP500_Close']  # Features to use
    look_back = 20  # Look-back period for sequence modeling

    for ticker in tickers:
        print(f"Processing {ticker}...")

        # Fetch and preprocess data for the current ticker
        df = fetch_data(ticker, start_date, end_date)
        X, y, scaler, scaled_df = preprocess_data(df, features, look_back)

        # Split data into training, validation, and test sets
        train_size = int(len(X) * 0.7)  # 70% for training
        val_size = int(len(X) * 0.15)  # 15% for validation

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        # Build and train the model for the current ticker
        input_shape = (X_train.shape[1], X_train.shape[2])
        model, history = build_and_train_model(X_train, y_train, X_val, y_val, input_shape)

        # Plot training history and save it
        plot_training_history(history, "evaluation_data", ticker)

        # Evaluate the model on test data
        evaluate_model(model, X_test, y_test, scaler, scaled_df, look_back, ticker)


if __name__ == "__main__":
    dependencies_file = "dependencies.txt"
    main_script = "main.py"
    install_dependencies(dependencies_file)

    # Run the main prediction process
    gather_predictions()
