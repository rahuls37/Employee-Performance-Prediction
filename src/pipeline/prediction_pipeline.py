import joblib
import pandas as pd

class PredictionPipeline:
    def __init__(self, model_path, encoders_path):
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoders_path)

    def preprocess(self, input_df):
        # Apply encoders to the input data
        for column, encoder in self.encoders.items():
            if column in input_df.columns:
                input_df[column] = encoder.transform(input_df[column])
        return input_df

    def predict(self, input_df):
        input_df = self.preprocess(input_df)
        prediction = self.model.predict(input_df)
        return prediction