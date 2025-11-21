import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import OUTPUTS_DIR

def pca_analysis(df, feature_names):
    """Performs PCA analysis."""
    print("\n" + "="*80)
    print("6. PCA ANALYSIS")
    print("="*80)
    
    X = df[feature_names]
    y = df['quality']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Full PCA
    pca_full = PCA()
    pca_full.fit(X_scaled)
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Plot Variance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(range(1, len(explained_variance) + 1), explained_variance * 100, alpha=0.7)
    axes[0].set_title('Scree Plot')
    axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'go-')
    axes[1].axhline(y=95, color='r', linestyle='--')
    axes[1].set_title('Cumulative Variance')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / '07_pca_variance.png', dpi=300)
    plt.close()
    
    # PCA Biplot
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='RdYlGn', alpha=0.6)
    plt.colorbar(scatter, label='Quality')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Biplot')
    plt.savefig(OUTPUTS_DIR / '08_pca_biplot.png', dpi=300)
    plt.close()
    print("✓ Plots saved: 07_pca_variance.png, 08_pca_biplot.png")

def train_quality_models(df, feature_names):
    """Trains and evaluates models."""
    print("\n" + "="*80)
    print("8. MODELING - QUALITY PREDICTION")
    print("="*80)
    
    X = df[feature_names]
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({'Model': name, 'MAE': mae, 'R2': r2})
        
    results_df = pd.DataFrame(results).sort_values('MAE')
    print("\nResults:")
    print(results_df)
    
    return results_df
