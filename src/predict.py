import joblib
import numpy as np
import pandas as pd
import os

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.le_education = None
        self.le_self_employed = None
        self.le_loan_status = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained model and associated files"""
        try:
            # Try loading from root models directory first
            model_paths = [
                ('models/model.pkl', 'models/scaler.pkl', 'models/feature_columns.pkl',
                 'models/le_education.pkl', 'models/le_self_employed.pkl', 'models/le_loan_status.pkl'),
                ('src/models/model.pkl', 'src/models/scaler.pkl', 'src/models/feature_columns.pkl',
                 'src/models/le_education.pkl', 'src/models/le_self_employed.pkl', 'src/models/le_loan_status.pkl')
            ]
            
            for paths in model_paths:
                try:
                    self.model = joblib.load(paths[0])
                    self.scaler = joblib.load(paths[1])
                    self.feature_columns = joblib.load(paths[2])
                    self.le_education = joblib.load(paths[3])
                    self.le_self_employed = joblib.load(paths[4])
                    self.le_loan_status = joblib.load(paths[5])
                    self.model_loaded = True
                    print(f"✅ Model loaded from: {paths[0]}")
                    return True
                except FileNotFoundError:
                    continue
            
            # If we get here, no model files were found
            raise FileNotFoundError("Model files not found in either 'models/' or 'src/models/' directories")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        if not self.model_loaded:
            raise Exception("Model not loaded. Call load_model() first.")
        
        # Create a copy of input data
        data = input_data.copy()
        
        # Encode categorical variables
        if 'education' in data:
            try:
                # Handle unseen categories
                if data['education'] in self.le_education.classes_:
                    data['education'] = self.le_education.transform([data['education']])[0]
                else:
                    # Use the most common category if unseen
                    data['education'] = 0
            except:
                data['education'] = 0
        
        if 'self_employed' in data:
            try:
                if data['self_employed'] in self.le_self_employed.classes_:
                    data['self_employed'] = self.le_self_employed.transform([data['self_employed']])[0]
                else:
                    data['self_employed'] = 0
            except:
                data['self_employed'] = 0
        
        # Create feature vector in the same order as training
        feature_vector = []
        for col in self.feature_columns:
            if col in data:
                feature_vector.append(data[col])
            else:
                feature_vector.append(0)  # Default value for missing features
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict(self, input_data):
        """Make prediction for loan approval"""
        if not self.model_loaded:
            if not self.load_model():
                return None, None
        
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            
            # Debug prints
            print("Raw prediction:", prediction)
            print("Inverse transformed:", self.le_loan_status.inverse_transform([prediction])[0])
            print("Probabilities:", prediction_proba)
            
            # Convert prediction back to original label
            prediction_label = self.le_loan_status.inverse_transform([prediction])[0]
            
            # Get confidence score
            confidence = max(prediction_proba) * 100
            
            return prediction_label, confidence
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None, None
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if not self.model_loaded:
            if not self.load_model():
                return None
        
        try:
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None

def check_model_files():
    """Check if all required model files exist"""
    # Check both possible locations
    locations = [
        ['models/model.pkl', 'models/scaler.pkl', 'models/feature_columns.pkl',
         'models/le_education.pkl', 'models/le_self_employed.pkl', 'models/le_loan_status.pkl'],
        ['src/models/model.pkl', 'src/models/scaler.pkl', 'src/models/feature_columns.pkl',
         'src/models/le_education.pkl', 'src/models/le_self_employed.pkl', 'src/models/le_loan_status.pkl']
    ]
    
    for required_files in locations:
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if len(missing_files) == 0:
            return True, []
    
    # If we get here, files are missing in both locations
    # Return the missing files from the preferred location (root models/)
    missing_files = []
    for file_path in locations[0]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return False, missing_files