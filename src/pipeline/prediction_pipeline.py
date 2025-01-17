import joblib
import pandas as pd
import numpy as np

class PredictionPipeline:
    def __init__(self, model_path, encoders_path):
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoders_path)
        
    def preprocess(self, input_df):
        try:
            # Create copy of input data
            df = input_df.copy()
            
            # Convert numeric columns
            numeric_cols = [
                'Age', 'DistanceFromHome', 'EmpEducationLevel', 
                'EmpEnvironmentSatisfaction', 'EmpHourlyRate', 
                'EmpJobInvolvement', 'EmpJobLevel', 'EmpJobSatisfaction',
                'NumCompaniesWorked', 'EmpLastSalaryHikePercent', 
                'EmpRelationshipSatisfaction', 'TotalWorkExperienceInYears', 
                'TrainingTimesLastYear', 'EmpWorkLifeBalance', 
                'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole',
                'YearsSinceLastPromotion', 'YearsWithCurrManager'
            ]
            
            # Convert categorical columns and encode them
            categorical_cols = [
                'Gender', 'EducationBackground', 'MaritalStatus',
                'EmpDepartment', 'BusinessTravelFrequency', 
                'OverTime', 'Attrition'
            ]
            
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
                
            for col in categorical_cols:
                if col in self.encoders:
                    df[col] = self.encoders[col].transform(df[col].values.reshape(-1, 1))
            
            # Convert to sequence format
            df['time_step_0'] = df['EmpLastSalaryHikePercent']
            df['time_step_1'] = df['EmpEnvironmentSatisfaction']
            
            return df[['time_step_0', 'time_step_1']]
            
        except Exception as e:
            raise Exception(f"Error in preprocessing: {str(e)}")

    def predict(self, input_df):
        try:
            # Preprocess the input data
            processed_df = self.preprocess(input_df)
            
            # Make prediction
            prediction = self.model.predict(processed_df)
            prediction = np.round(prediction).astype(int)
            return prediction
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")