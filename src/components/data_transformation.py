import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from pathlib import Path
import joblib
from src import logger
from typing import Tuple
from src.entity.config_entity import DataTransformationConfig
from src.utils.common import create_directories


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.encoders = {}
        create_directories([self.config.root_dir])

    def clean_and_encode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            # Convert numeric columns to appropriate types
            numeric_cols = [
                'Age', 'DistanceFromHome', 'EmpEducationLevel', 
                'EmpEnvironmentSatisfaction', 'EmpHourlyRate', 
                'EmpJobInvolvement', 'EmpJobLevel', 'EmpJobSatisfaction',
                'NumCompaniesWorked', 'EmpLastSalaryHikePercent', 
                'EmpRelationshipSatisfaction', 'TotalWorkExperienceInYears', 
                'TrainingTimesLastYear', 'EmpWorkLifeBalance', 
                'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole',
                'YearsSinceLastPromotion', 'YearsWithCurrManager', 
                'PerformanceRating'
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Encode categorical columns
            categorical_cols = ['Gender', 'EducationBackground', 'MaritalStatus', 
                                 'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency', 
                                 'OverTime', 'Attrition']

            for col in categorical_cols:
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
                self.encoders[f'{col}_encoder'] = label_encoder

            # Fill missing numeric values with median
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

            logger.info("Data cleaning and encoding completed successfully")
            return df
        except Exception as e:
            logger.error(f"Error in clean_and_encode_data: {e}")
            raise e

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.config.seq_length):
            sequence = data[i:i + self.config.seq_length + 1]
            if not np.any(np.isnan(sequence)):
                X.append(sequence[:-1])
                y.append(sequence[-1])
        return np.array(X), np.array(y)

    def save_data(self, X: np.ndarray, y: np.ndarray, filename: str) -> pd.DataFrame:
        """
        Save the transformed data into a CSV file.

        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Target data.
            filename (str): Name of the file to save the data.

        Returns:
            pd.DataFrame: Combined DataFrame of features and target.
        """
        try:
            # Reshape feature data for saving as 2D DataFrame
            X_2d = X.reshape(X.shape[0], -1)
            feature_cols = [f'time_step_{i}' for i in range(X.shape[1])]

            # Create DataFrame for features and target
            X_df = pd.DataFrame(X_2d, columns=feature_cols)
            y_df = pd.DataFrame(y, columns=['target'])

            combined_df = pd.concat([X_df, y_df], axis=1)

            # Save to file
            save_path = Path(self.config.root_dir) / filename
            combined_df.to_csv(save_path, index=False)

            logger.info(f"Data saved to: {save_path}")
            return combined_df
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise e

    def train_test_spliting(self):
        try:
            logger.info("Started data transformation")

            # Read the data from CSV path defined in config
            data = pd.read_csv(self.config.data_path)
            logger.info(f"Read data from {self.config.data_path}, shape: {data.shape}")

            # Clean and encode the data
            processed_df = self.clean_and_encode_data(data)
            target_data = processed_df['PerformanceRating'].values

            # Create sequences for the target data
            X, y = self.create_sequences(target_data)

            if len(X) == 0:
                raise ValueError("No valid sequences could be created. Check data integrity.")

            logger.info(f"Created sequences with shape X: {X.shape}, y: {y.shape}")

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            # Save train and test data to CSV files
            train_df = self.save_data(X_train, y_train, "train.csv")
            test_df = self.save_data(X_test, y_test, "test.csv")

            # Save encoders using joblib
            encoder_path = Path(self.config.root_dir) / "encoders.joblib"
            joblib.dump(self.encoders, encoder_path)

            logger.info("Data transformation completed")
            logger.info(f"Training set shape: {train_df.shape}")
            logger.info(f"Test set shape: {test_df.shape}")

            return train_df, test_df

        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e
