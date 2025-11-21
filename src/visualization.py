import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.config import OUTPUTS_DIR, PLOT_SETTINGS

# Apply plot settings
plt.style.use(PLOT_SETTINGS['style'])
sns.set_palette(PLOT_SETTINGS['palette'])
plt.rcParams['figure.figsize'] = PLOT_SETTINGS['figsize']
plt.rcParams['font.size'] = PLOT_SETTINGS['fontsize']
plt.rcParams['axes.titlesize'] = PLOT_SETTINGS['titlesize']
plt.rcParams['axes.labelsize'] = PLOT_SETTINGS['labelsize']

def plot_distributions(df, feature_names):
    """Plots distributions of all features."""
    print("\n--- 3.1 Feature Distributions ---")
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_names):
        if idx < len(axes):
            axes[idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution: {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
            
            mean_val = df[col].mean()
            median_val = df[col].median()
            axes[idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[idx].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / '01_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot saved: 01_feature_distributions.png")

def plot_quality_distribution(df):
    """Plots quality distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # General
    df['quality'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Quality Distribution - General', fontweight='bold')
    
    # Red
    df[df['type'] == 'red']['quality'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='darkred', edgecolor='black')
    axes[1].set_title('Quality Distribution - Red Wine', fontweight='bold')
    
    # White
    df[df['type'] == 'white']['quality'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='gold', edgecolor='black')
    axes[2].set_title('Quality Distribution - White Wine', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / '02_quality_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot saved: 02_quality_distribution.png")

def plot_outliers(df, feature_names):
    """Plots boxplots for outliers."""
    print("\n--- 3.2 Outlier Analysis ---")
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_names):
        if idx < len(axes):
            bp = axes[idx].boxplot([df[col]], labels=[col], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            axes[idx].set_title(f'Boxplot: {col}', fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / '03_outliers_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot saved: 03_outliers_boxplots.png")

def plot_correlation_matrix(df, feature_names):
    """Plots correlation matrix."""
    print("\n--- Correlation Matrix ---")
    
    df_numeric = df[feature_names + ['quality']].copy()
    corr_matrix = df_numeric.corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1, ax=ax)
    ax.set_title('Correlation Matrix', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / '05_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot saved: 05_correlation_matrix.png")
