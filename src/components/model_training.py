import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src import logger
from src.entity.config_entity import (ModelTrainerConfig)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            # Ensure the directory for saving the model exists
            model_dir = self.config.model_path
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)  # Create directory if it doesn't exist

            # Load the training and testing data
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            logger.info(f"Training data loaded from {self.config.train_data_path}, shape: {train_data.shape}")
            logger.info(f"Testing data loaded from {self.config.test_data_path}, shape: {test_data.shape}")

            # Split the training data into features (X) and target (y)
            X_train = train_data.drop(columns=[self.config.target_column])
            y_train = train_data[self.config.target_column]
            
            # Split the test data into features (X) and target (y)
            X_test = test_data.drop(columns=[self.config.target_column])
            y_test = test_data[self.config.target_column]

            # Initialize the Linear Regression model
            model = LinearRegression()

            # Train the model
            model.fit(X_train, y_train)
            logger.info("Model training completed successfully.")

            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            logger.info(f"Model Mean Squared Error: {mse}")

            # Save the trained model to the specified path
            model_save_path = os.path.join(model_dir, 'linear_regression_model.joblib')
            joblib.dump(model, model_save_path)
            logger.info(f"Model saved at {model_save_path}")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise e
