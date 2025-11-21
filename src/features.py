import pandas as pd

def feature_engineering(df, feature_names):
    """
    Creates new features based on enological knowledge.
    """
    print("\n" + "="*80)
    print("7. FEATURE ENGINEERING")
    print("="*80)
    
    # 1. Total Acidity
    df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
    
    # 2. Free/Total SO2 Ratio
    df['free_so2_ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
    
    # 3. Adjusted Density
    df['density_adjusted'] = df['density'] * df['alcohol']
    
    # 4. Acidity Index
    df['acidity_index'] = df['total_acidity'] / df['pH']
    
    # 5. Sugar/Alcohol Ratio
    df['sugar_alcohol_ratio'] = df['residual sugar'] / df['alcohol']
    
    # 6. Adjusted Chlorides
    df['chlorides_adjusted'] = df['chlorides'] * df['pH']
    
    # 7. Quality Class
    df['quality_class'] = pd.cut(df['quality'], bins=[0, 5, 6, 10], labels=['Low', 'Medium', 'High'])
    
    new_features = ['total_acidity', 'free_so2_ratio', 'density_adjusted', 
                   'acidity_index', 'sugar_alcohol_ratio', 'chlorides_adjusted']
    
    print(f"\n✓ Created {len(new_features)} new features.")
    
    return df, feature_names + new_features
