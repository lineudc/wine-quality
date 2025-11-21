import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, mannwhitneyu
from src.visualization import plot_distributions, plot_quality_distribution, plot_outliers, plot_correlation_matrix
from src.config import OUTPUTS_DIR

def preliminary_analysis(df):
    """Performs preliminary data analysis."""
    print("\n" + "="*80)
    print("2. PRELIMINARY DATA ANALYSIS")
    print("="*80)
    
    print("\n--- General Info ---")
    print(df.info())
    
    print("\n--- First Rows ---")
    print(df.head(10))
    
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    
    # Missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found!")
    else:
        print(f"⚠ Missing values found:\n{missing[missing > 0]}")
    
    # Duplicates
    print("\n--- Duplicates ---")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠ Found {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)")
        print("Suggestion: Remove duplicates to avoid bias.")
        df = df.drop_duplicates()
        print(f"✓ Duplicates removed. New size: {len(df)} samples")
    else:
        print("✓ No duplicates found!")
        
    return df

def exploratory_data_analysis(df, feature_names):
    """Performs full EDA."""
    print("\n" + "="*80)
    print("3. EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    plot_distributions(df, feature_names)
    plot_quality_distribution(df)
    plot_outliers(df, feature_names)
    test_normality(df, feature_names)

def test_normality(df, feature_names):
    """Tests normality of distributions."""
    print("\n--- 3.3 Normality Tests ---")
    
    normality_results = {}
    
    for col in feature_names:
        # Shapiro-Wilk (limit to 5000 samples for performance)
        if len(df) < 5000:
            stat_shapiro, p_shapiro = shapiro(df[col].sample(min(5000, len(df))))
        else:
            stat_shapiro, p_shapiro = None, None
        
        # D'Agostino's K² Test
        stat_dagostino, p_dagostino = normaltest(df[col])
        
        normality_results[col] = {
            'Shapiro-Wilk p-value': p_shapiro,
            "D'Agostino p-value": p_dagostino,
            'Normal (α=0.05)': 'Yes' if (p_dagostino > 0.05) else 'No'
        }
    
    normality_df = pd.DataFrame(normality_results).T
    print("\nNormality Test Results:")
    print(normality_df)

def comparative_analysis(df, feature_names):
    """Compares red vs white wines."""
    print("\n" + "="*80)
    print("4. COMPARATIVE ANALYSIS: RED vs WHITE")
    print("="*80)
    
    df_red = df[df['type'] == 'red']
    df_white = df[df['type'] == 'white']
    
    comparison = pd.DataFrame({
        'Red - Mean': df_red[feature_names].mean(),
        'Red - Std': df_red[feature_names].std(),
        'White - Mean': df_white[feature_names].mean(),
        'White - Std': df_white[feature_names].std(),
    })
    
    comparison['Diff (%)'] = ((comparison['White - Mean'] - comparison['Red - Mean']) / 
                                   comparison['Red - Mean'] * 100)
    
    print(comparison.round(3))
    
    # Violin plots
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_names):
        if idx < len(axes):
            data_to_plot = [df_red[col], df_white[col]]
            parts = axes[idx].violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
            
            colors = ['darkred', 'gold']
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            axes[idx].set_title(f'{col}', fontweight='bold')
            axes[idx].set_xticks([1, 2])
            axes[idx].set_xticklabels(['Red', 'White'])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Mann-Whitney U test
            stat, p_value = mannwhitneyu(df_red[col], df_white[col])
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            axes[idx].text(0.5, 0.95, f'p: {p_value:.4f} {sig}', transform=axes[idx].transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
            
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / '04_red_vs_white_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot saved: 04_red_vs_white_comparison.png")

def correlation_analysis(df, feature_names):
    """Performs correlation analysis."""
    print("\n" + "="*80)
    print("5. CORRELATION ANALYSIS")
    print("="*80)
    
    plot_correlation_matrix(df, feature_names)
    
    # Correlation with quality
    corr_matrix = df[feature_names + ['quality']].corr()
    quality_corr = corr_matrix['quality'].drop('quality').sort_values(ascending=False)
    
    print("\n--- Correlation with Quality ---")
    print(quality_corr)
    
    # Plot quality correlations
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    quality_corr.plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Correlation with Quality - General')
    
    df_red = df[df['type'] == 'red']
    corr_red = df_red[feature_names + ['quality']].corr()['quality'].drop('quality').sort_values(ascending=False)
    corr_red.plot(kind='barh', ax=axes[1], color='darkred', edgecolor='black')
    axes[1].set_title('Correlation with Quality - Red')
    
    df_white = df[df['type'] == 'white']
    corr_white = df_white[feature_names + ['quality']].corr()['quality'].drop('quality').sort_values(ascending=False)
    corr_white.plot(kind='barh', ax=axes[2], color='gold', edgecolor='black')
    axes[2].set_title('Correlation with Quality - White')
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / '06_quality_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot saved: 06_quality_correlations.png")
