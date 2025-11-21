import sys
from src.data_loader import load_and_combine_data
from src.eda import preliminary_analysis, exploratory_data_analysis, comparative_analysis, correlation_analysis
from src.features import feature_engineering
from src.models import pca_analysis, train_quality_models

def main():
    print("Starting Wine Quality Analysis...")
    
    # 1. Load Data
    df = load_and_combine_data()
    
    # Feature names (initial)
    feature_names = [col for col in df.columns if col not in ['quality', 'type']]
    
    # 2. Preliminary Analysis
    df = preliminary_analysis(df)
    
    # 3. EDA
    exploratory_data_analysis(df, feature_names)
    
    # 4. Comparative Analysis
    comparative_analysis(df, feature_names)
    
    # 5. Correlation Analysis
    correlation_analysis(df, feature_names)
    
    # 6. PCA
    pca_analysis(df, feature_names)
    
    # 7. Feature Engineering
    df, feature_names = feature_engineering(df, feature_names)
    
    # 8. Modeling
    train_quality_models(df, feature_names)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
