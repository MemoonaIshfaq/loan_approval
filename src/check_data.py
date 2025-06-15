import pandas as pd
import sys

def check_data(file_path):
    """Check the structure of the dataset"""
    try:
        print("🔍 Checking dataset structure...")
        df = pd.read_csv(file_path)
        
        print(f"\n📊 Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")
        
        print(f"\n📋 Original Column Names:")
        for i, col in enumerate(df.columns):
            print(f"{i+1:2d}. '{col}'")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        print(f"\n✨ Cleaned Column Names:")
        for i, col in enumerate(df.columns):
            print(f"{i+1:2d}. '{col}'")
        
        print(f"\n📈 Data Types:")
        print(df.dtypes)
        
        print(f"\n🎯 Target Variable (loan_status) Values:")
        if 'loan_status' in df.columns:
            print(df['loan_status'].value_counts())
            print(f"Unique values: {df['loan_status'].unique()}")
        else:
            print("❌ 'loan_status' column not found!")
        
        print(f"\n🔍 Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("✅ No missing values found!")
        
        print(f"\n📊 Sample Data (first 3 rows):")
        print(df.head(3))
        
        return True
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please make sure the file exists in the correct location.")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if file path is provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/loan_data.csv"
    
    print(f"📁 Checking file: {file_path}")
    check_data(file_path)