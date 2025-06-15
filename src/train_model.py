import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_and_preprocess_data(file_path):
    """Load and preprocess the loan dataset"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Clean column names by removing leading/trailing spaces
    df.columns = df.columns.str.strip()
    
    # Clean string columns by removing leading/trailing spaces
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic info
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Show unique values for categorical columns
    print("\nUnique values in categorical columns:")
    for col in ['education', 'self_employed', 'loan_status']:
        if col in df.columns:
            print(f"{col}: {df[col].unique()}")
    
    # Handle missing values
    # Fill numerical columns with median
    numerical_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
                     'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                     'luxury_assets_value', 'bank_asset_value']
    
    for col in numerical_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = ['education', 'self_employed']
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    le_education = LabelEncoder()
    le_self_employed = LabelEncoder()
    le_loan_status = LabelEncoder()
    
    if 'education' in df.columns:
        df['education'] = le_education.fit_transform(df['education'].astype(str))
    
    if 'self_employed' in df.columns:
        df['self_employed'] = le_self_employed.fit_transform(df['self_employed'].astype(str))
    
    # Encode target variable
    if 'loan_status' in df.columns:
        df['loan_status'] = le_loan_status.fit_transform(df['loan_status'].astype(str))
        print(f"\nLoan status classes: {le_loan_status.classes_}")
    else:
        raise ValueError("Column 'loan_status' not found in dataset")
    
    print("\nAfter preprocessing - Missing values:")
    print(df.isnull().sum())
    
    return df, le_education, le_self_employed, le_loan_status

def prepare_features(df):
    """Prepare features for training"""
    # Drop loan_id as it's not useful for prediction
    feature_columns = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
                      'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                      'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
    
    # Only use columns that exist in the dataset
    available_features = [col for col in feature_columns if col in df.columns]
    
    if not available_features:
        raise ValueError("No feature columns found in dataset")
    
    if 'loan_status' not in df.columns:
        raise ValueError("Target column 'loan_status' not found in dataset")
    
    X = df[available_features]
    y = df['loan_status']
    
    print(f"\nFeatures used: {available_features}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Target variable distribution:")
    print(y.value_counts())
    
    return X, y, available_features

def train_model(X, y):
    """Train the loan prediction model"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, scaler, accuracy

def save_model_and_scaler(model, scaler, model_path='models/model.pkl', scaler_path='models/scaler.pkl'):
    """Save the trained model and scaler"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

def main():
    """Main training pipeline"""
    try:
        # Load and preprocess data
        df, le_education, le_self_employed, le_loan_status = load_and_preprocess_data('data/loan_data.csv')
        
        # Prepare features
        X, y, feature_columns = prepare_features(df)
        
        # Train model
        model, scaler, accuracy = train_model(X, y)
        
        # Save model and scaler
        save_model_and_scaler(model, scaler)
        
        # Save feature columns and encoders for later use
        joblib.dump(feature_columns, 'models/feature_columns.pkl')
        joblib.dump(le_education, 'models/le_education.pkl')
        joblib.dump(le_self_employed, 'models/le_self_employed.pkl')
        joblib.dump(le_loan_status, 'models/le_loan_status.pkl')
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Final Model Accuracy: {accuracy:.4f}")
        print(f"💾 All files saved in 'models/' directory")
        
    except FileNotFoundError:
        print("❌ Error: Could not find 'data/loan_data.csv'")
        print("Please make sure your dataset is placed in the 'data/' directory with the name 'loan_data.csv'")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")

if __name__ == "__main__":
    main()