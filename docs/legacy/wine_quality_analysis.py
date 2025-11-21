"""
===============================================================================
WINE QUALITY ANALYSIS - Vinho Verde Dataset
===============================================================================
Análise Exploratória de Dados e Machine Learning para o Dataset Wine Quality

Autor: Análise para Lineu
Dataset: Cortez et al., 2009 - UCI Wine Quality Dataset
Objetivo: EDA completa, ML para predição de qualidade e classificação de tipos
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, roc_curve)

# Configurações visuais
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Configurações do pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 3)


class WineQualityAnalyzer:
    """
    Classe principal para análise do Wine Quality Dataset
    """
    
    def __init__(self, red_wine_path, white_wine_path):
        """
        Inicializa o analisador com os caminhos dos datasets
        
        Parameters:
        -----------
        red_wine_path : str
            Caminho para o arquivo CSV de vinhos tintos
        white_wine_path : str
            Caminho para o arquivo CSV de vinhos brancos
        """
        self.red_wine_path = red_wine_path
        self.white_wine_path = white_wine_path
        self.df_combined = None
        self.df_red = None
        self.df_white = None
        self.feature_names = None
        self.models_quality = {}
        self.models_type = {}
        self.scaler = None
        
    def load_and_combine_data(self):
        """
        Carrega e combina os datasets de vinho tinto e branco
        """
        print("="*80)
        print("1. CARREGAMENTO E COMBINAÇÃO DOS DADOS")
        print("="*80)
        
        # Carregar dados
        self.df_red = pd.read_csv(self.red_wine_path, sep=';')
        self.df_white = pd.read_csv(self.white_wine_path, sep=';')
        
        # Adicionar coluna de tipo
        self.df_red['type'] = 'red'
        self.df_white['type'] = 'white'
        
        # Combinar datasets
        self.df_combined = pd.concat([self.df_red, self.df_white], axis=0, ignore_index=True)
        
        # Armazenar nomes das features
        self.feature_names = [col for col in self.df_combined.columns 
                             if col not in ['quality', 'type']]
        
        print(f"\n✓ Vinho Tinto: {len(self.df_red)} amostras")
        print(f"✓ Vinho Branco: {len(self.df_white)} amostras")
        print(f"✓ Dataset Combinado: {len(self.df_combined)} amostras")
        print(f"✓ Features: {len(self.feature_names)}")
        print(f"\nFeatures disponíveis:\n{self.feature_names}")
        
        return self.df_combined
    
    def preliminary_analysis(self):
        """
        Análise preliminar dos dados: estrutura, tipos, missing values, duplicatas
        """
        print("\n" + "="*80)
        print("2. ANÁLISE PRELIMINAR DOS DADOS")
        print("="*80)
        
        print("\n--- Informações Gerais do Dataset ---")
        print(self.df_combined.info())
        
        print("\n--- Primeiras Linhas ---")
        print(self.df_combined.head(10))
        
        print("\n--- Estatísticas Descritivas ---")
        print(self.df_combined.describe())
        
        # Verificar valores ausentes
        print("\n--- Verificação de Valores Ausentes ---")
        missing = self.df_combined.isnull().sum()
        if missing.sum() == 0:
            print("✓ Não há valores ausentes no dataset!")
        else:
            print(f"⚠ Valores ausentes encontrados:\n{missing[missing > 0]}")
        
        # Verificar duplicatas
        print("\n--- Verificação de Duplicatas ---")
        duplicates = self.df_combined.duplicated().sum()
        if duplicates > 0:
            print(f"⚠ Encontradas {duplicates} linhas duplicadas ({duplicates/len(self.df_combined)*100:.2f}%)")
            print("Sugestão: Remover duplicatas para evitar viés nos modelos")
            
            # Remover duplicatas
            self.df_combined = self.df_combined.drop_duplicates()
            print(f"✓ Duplicatas removidas. Novo tamanho: {len(self.df_combined)} amostras")
        else:
            print("✓ Não há duplicatas no dataset!")
        
        # Distribuição da variável target
        print("\n--- Distribuição da Variável Target (Quality) ---")
        print(self.df_combined['quality'].value_counts().sort_index())
        
        # Distribuição por tipo
        print("\n--- Distribuição por Tipo de Vinho ---")
        print(self.df_combined['type'].value_counts())
        
    def exploratory_data_analysis(self):
        """
        Análise Exploratória de Dados (EDA) completa com visualizações
        """
        print("\n" + "="*80)
        print("3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
        print("="*80)
        
        # 3.1 Distribuições das variáveis
        self._plot_distributions()
        
        # 3.2 Análise de outliers
        self._analyze_outliers()
        
        # 3.3 Testes de normalidade
        self._test_normality()
        
    def _plot_distributions(self):
        """
        Plota distribuições de todas as features
        """
        print("\n--- 3.1 Distribuições das Features ---")
        
        # Distribuições das features físico-químicas
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.ravel()
        
        for idx, col in enumerate(self.feature_names):
            axes[idx].hist(self.df_combined[col], bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribuição: {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequência')
            axes[idx].grid(True, alpha=0.3)
            
            # Adicionar estatísticas
            mean_val = self.df_combined[col].mean()
            median_val = self.df_combined[col].median()
            axes[idx].axvline(mean_val, color='red', linestyle='--', 
                            label=f'Média: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='green', linestyle='--', 
                            label=f'Mediana: {median_val:.2f}')
            axes[idx].legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/01_feature_distributions.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 01_feature_distributions.png")
        
        # Distribuição da qualidade
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Geral
        self.df_combined['quality'].value_counts().sort_index().plot(kind='bar', 
                                                                      ax=axes[0], 
                                                                      color='steelblue', 
                                                                      edgecolor='black')
        axes[0].set_title('Distribuição da Qualidade - Geral', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('Qualidade (pontos)')
        axes[0].set_ylabel('Frequência')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Vinho Tinto
        self.df_red['quality'].value_counts().sort_index().plot(kind='bar', 
                                                                 ax=axes[1], 
                                                                 color='darkred', 
                                                                 edgecolor='black')
        axes[1].set_title('Distribuição da Qualidade - Vinho Tinto', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('Qualidade (pontos)')
        axes[1].set_ylabel('Frequência')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Vinho Branco
        self.df_white['quality'].value_counts().sort_index().plot(kind='bar', 
                                                                   ax=axes[2], 
                                                                   color='gold', 
                                                                   edgecolor='black')
        axes[2].set_title('Distribuição da Qualidade - Vinho Branco', fontweight='bold', fontsize=14)
        axes[2].set_xlabel('Qualidade (pontos)')
        axes[2].set_ylabel('Frequência')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/02_quality_distribution.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 02_quality_distribution.png")
        
    def _analyze_outliers(self):
        """
        Analisa outliers usando boxplots e método IQR
        """
        print("\n--- 3.2 Análise de Outliers ---")
        
        # Boxplots das features
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.ravel()
        
        outliers_summary = {}
        
        for idx, col in enumerate(self.feature_names):
            # Boxplot
            bp = axes[idx].boxplot([self.df_combined[col]], labels=[col], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_edgecolor('black')
            axes[idx].set_title(f'Boxplot: {col}', fontweight='bold')
            axes[idx].set_ylabel('Valor')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Calcular outliers (método IQR)
            Q1 = self.df_combined[col].quantile(0.25)
            Q3 = self.df_combined[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df_combined[(self.df_combined[col] < lower_bound) | 
                                       (self.df_combined[col] > upper_bound)]
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.df_combined) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/03_outliers_boxplots.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 03_outliers_boxplots.png")
        
        # Resumo de outliers
        print("\nResumo de Outliers (método IQR):")
        print("-" * 80)
        outliers_df = pd.DataFrame(outliers_summary).T
        outliers_df = outliers_df.sort_values('percentage', ascending=False)
        print(outliers_df)
        
        print("\n⚠ Interpretação:")
        print("- Outliers podem ser valores legítimos ou erros de medição")
        print("- Para vinhos, variação natural é esperada")
        print("- Sugestão: Usar RobustScaler ao invés de StandardScaler para ML")
        
    def _test_normality(self):
        """
        Testa normalidade das distribuições usando múltiplos testes
        """
        print("\n--- 3.3 Testes de Normalidade ---")
        
        normality_results = {}
        
        for col in self.feature_names:
            # Shapiro-Wilk Test (melhor para n < 5000)
            if len(self.df_combined) < 5000:
                stat_shapiro, p_shapiro = shapiro(self.df_combined[col].sample(min(5000, len(self.df_combined))))
            else:
                stat_shapiro, p_shapiro = None, None
            
            # D'Agostino's K² Test
            stat_dagostino, p_dagostino = normaltest(self.df_combined[col])
            
            normality_results[col] = {
                'Shapiro-Wilk p-value': p_shapiro,
                "D'Agostino p-value": p_dagostino,
                'Normal (α=0.05)': 'Sim' if (p_dagostino > 0.05) else 'Não'
            }
        
        normality_df = pd.DataFrame(normality_results).T
        print("\nResultados dos Testes de Normalidade:")
        print("-" * 80)
        print(normality_df)
        
        normal_count = (normality_df['Normal (α=0.05)'] == 'Sim').sum()
        print(f"\n✓ {normal_count}/{len(self.feature_names)} features seguem distribuição aproximadamente normal")
        
        if normal_count < len(self.feature_names) / 2:
            print("⚠ Maioria das features NÃO segue distribuição normal")
            print("→ Sugestão: Considerar transformações (log, Box-Cox) ou modelos não-paramétricos")
        
    def comparative_analysis(self):
        """
        Análise comparativa entre vinhos tintos e brancos
        """
        print("\n" + "="*80)
        print("4. ANÁLISE COMPARATIVA: VINHOS TINTOS vs BRANCOS")
        print("="*80)
        
        # Estatísticas comparativas
        print("\n--- Estatísticas Comparativas ---")
        
        comparison = pd.DataFrame({
            'Tinto - Média': self.df_red[self.feature_names].mean(),
            'Tinto - Desvio': self.df_red[self.feature_names].std(),
            'Branco - Média': self.df_white[self.feature_names].mean(),
            'Branco - Desvio': self.df_white[self.feature_names].std(),
        })
        
        comparison['Diferença (%)'] = ((comparison['Branco - Média'] - comparison['Tinto - Média']) / 
                                       comparison['Tinto - Média'] * 100)
        
        print(comparison.round(3))
        
        # Identificar maiores diferenças
        print("\n--- Features com Maiores Diferenças entre Tipos ---")
        top_diff = comparison['Diferença (%)'].abs().sort_values(ascending=False).head(5)
        print(top_diff)
        
        # Violin plots comparativos
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        axes = axes.ravel()
        
        for idx, col in enumerate(self.feature_names):
            data_to_plot = [self.df_red[col], self.df_white[col]]
            parts = axes[idx].violinplot(data_to_plot, positions=[1, 2], 
                                        showmeans=True, showmedians=True)
            
            # Colorir violinos
            colors = ['darkred', 'gold']
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            axes[idx].set_title(f'{col}', fontweight='bold')
            axes[idx].set_xticks([1, 2])
            axes[idx].set_xticklabels(['Tinto', 'Branco'])
            axes[idx].set_ylabel('Valor')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Adicionar teste estatístico (Mann-Whitney U)
            from scipy.stats import mannwhitneyu
            statistic, p_value = mannwhitneyu(self.df_red[col], self.df_white[col])
            significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            axes[idx].text(0.5, 0.95, f'p-value: {p_value:.4f} {significance}',
                          transform=axes[idx].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=8)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/04_red_vs_white_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Gráfico salvo: 04_red_vs_white_comparison.png")
        
        # Análise da qualidade por tipo
        print("\n--- Qualidade Média por Tipo ---")
        quality_comparison = pd.DataFrame({
            'Vinho Tinto': [
                self.df_red['quality'].mean(),
                self.df_red['quality'].median(),
                self.df_red['quality'].std()
            ],
            'Vinho Branco': [
                self.df_white['quality'].mean(),
                self.df_white['quality'].median(),
                self.df_white['quality'].std()
            ]
        }, index=['Média', 'Mediana', 'Desvio Padrão'])
        
        print(quality_comparison.round(3))
        
    def correlation_analysis(self):
        """
        Análise de correlação entre features e com o target
        """
        print("\n" + "="*80)
        print("5. ANÁLISE DE CORRELAÇÃO")
        print("="*80)
        
        # Matriz de correlação
        print("\n--- Matriz de Correlação Completa ---")
        
        # Preparar dados numéricos
        df_numeric = self.df_combined[self.feature_names + ['quality']].copy()
        
        # Calcular correlação
        corr_matrix = df_numeric.corr()
        
        # Heatmap geral
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Matriz de Correlação - Dataset Completo', 
                    fontweight='bold', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/05_correlation_matrix.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 05_correlation_matrix.png")
        
        # Correlação com qualidade
        print("\n--- Correlação das Features com Qualidade ---")
        quality_corr = corr_matrix['quality'].drop('quality').sort_values(ascending=False)
        print(quality_corr)
        
        # Plot das correlações com qualidade
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Geral
        quality_corr.plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='black')
        axes[0].set_title('Correlação com Qualidade - Geral', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('Correlação de Pearson')
        axes[0].axvline(0, color='black', linewidth=0.8)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Vinho Tinto
        corr_red = self.df_red[self.feature_names + ['quality']].corr()['quality'].drop('quality').sort_values(ascending=False)
        corr_red.plot(kind='barh', ax=axes[1], color='darkred', edgecolor='black')
        axes[1].set_title('Correlação com Qualidade - Vinho Tinto', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('Correlação de Pearson')
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Vinho Branco
        corr_white = self.df_white[self.feature_names + ['quality']].corr()['quality'].drop('quality').sort_values(ascending=False)
        corr_white.plot(kind='barh', ax=axes[2], color='gold', edgecolor='black')
        axes[2].set_title('Correlação com Qualidade - Vinho Branco', fontweight='bold', fontsize=14)
        axes[2].set_xlabel('Correlação de Pearson')
        axes[2].axvline(0, color='black', linewidth=0.8)
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/06_quality_correlations.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 06_quality_correlations.png")
        
        # Identificar multicolinearidade
        print("\n--- Análise de Multicolinearidade ---")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7 and corr_matrix.columns[i] != 'quality' and corr_matrix.columns[j] != 'quality':
                    high_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlação': corr_matrix.iloc[i, j]
                    })
        
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr).sort_values('Correlação', 
                                                               key=abs, 
                                                               ascending=False)
            print("\n⚠ Pares de features com alta correlação (|r| > 0.7):")
            print(high_corr_df.to_string(index=False))
            print("\n→ Sugestão: Considerar remover uma das features de cada par ou usar PCA")
        else:
            print("✓ Não há pares de features com correlação muito alta (|r| > 0.7)")
    
    def pca_analysis(self):
        """
        Análise de Componentes Principais (PCA)
        """
        print("\n" + "="*80)
        print("6. ANÁLISE PCA (PRINCIPAL COMPONENT ANALYSIS)")
        print("="*80)
        
        # Preparar dados
        X = self.df_combined[self.feature_names]
        y = self.df_combined['quality']
        wine_type = self.df_combined['type']
        
        # Padronizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA completo
        pca_full = PCA()
        X_pca_full = pca_full.fit_transform(X_scaled)
        
        # Variância explicada
        explained_variance = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print("\n--- Variância Explicada por Componente ---")
        pca_var_df = pd.DataFrame({
            'Componente': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Variância Explicada (%)': explained_variance * 100,
            'Variância Acumulada (%)': cumulative_variance * 100
        })
        print(pca_var_df.to_string(index=False))
        
        # Determinar número ótimo de componentes (>= 95% variância)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"\n✓ Número de componentes para 95% da variância: {n_components_95}")
        
        # Plot da variância explicada
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scree plot
        axes[0].bar(range(1, len(explained_variance) + 1), explained_variance * 100, 
                   alpha=0.7, edgecolor='black', color='steelblue')
        axes[0].plot(range(1, len(explained_variance) + 1), explained_variance * 100, 
                    'ro-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Componente Principal', fontsize=12)
        axes[0].set_ylabel('Variância Explicada (%)', fontsize=12)
        axes[0].set_title('Scree Plot - Variância Explicada por Componente', 
                         fontweight='bold', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Variância acumulada
        axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 
                    'go-', linewidth=2, markersize=8)
        axes[1].axhline(y=95, color='r', linestyle='--', linewidth=2, label='95% Threshold')
        axes[1].axvline(x=n_components_95, color='r', linestyle='--', linewidth=2, alpha=0.5)
        axes[1].set_xlabel('Número de Componentes', fontsize=12)
        axes[1].set_ylabel('Variância Acumulada (%)', fontsize=12)
        axes[1].set_title('Variância Acumulada', fontweight='bold', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/07_pca_variance_explained.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 07_pca_variance_explained.png")
        
        # PCA Biplot (PC1 vs PC2)
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Colorido por qualidade
        scatter1 = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                                  c=y, cmap='RdYlGn', alpha=0.6, 
                                  edgecolor='black', linewidth=0.5, s=50)
        axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        axes[0].set_title('PCA Biplot - Colorido por Qualidade', fontweight='bold', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Qualidade')
        
        # Colorido por tipo
        colors = ['darkred' if t == 'red' else 'gold' for t in wine_type]
        axes[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                       c=colors, alpha=0.6, edgecolor='black', linewidth=0.5, s=50)
        axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        axes[1].set_title('PCA Biplot - Colorido por Tipo de Vinho', fontweight='bold', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # Legenda customizada
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='darkred', edgecolor='black', label='Tinto'),
                          Patch(facecolor='gold', edgecolor='black', label='Branco')]
        axes[1].legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/08_pca_biplot.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 08_pca_biplot.png")
        
        # Loadings (contribuição das features)
        print("\n--- Loadings das Features nos Componentes Principais ---")
        loadings = pd.DataFrame(
            pca_2d.components_.T,
            columns=['PC1', 'PC2'],
            index=self.feature_names
        )
        loadings['PC1_abs'] = np.abs(loadings['PC1'])
        loadings['PC2_abs'] = np.abs(loadings['PC2'])
        
        print("\nTop 5 features que mais contribuem para PC1:")
        print(loadings.sort_values('PC1_abs', ascending=False)[['PC1']].head())
        
        print("\nTop 5 features que mais contribuem para PC2:")
        print(loadings.sort_values('PC2_abs', ascending=False)[['PC2']].head())
        
    def feature_engineering(self):
        """
        Criação de novas features baseadas em conhecimento enológico
        """
        print("\n" + "="*80)
        print("7. FEATURE ENGINEERING")
        print("="*80)
        
        print("\n--- Criando Novas Features ---")
        
        # 1. Acidez Total
        self.df_combined['total_acidity'] = (self.df_combined['fixed acidity'] + 
                                            self.df_combined['volatile acidity'])
        print("✓ total_acidity = fixed acidity + volatile acidity")
        
        # 2. Razão SO2 livre/total
        self.df_combined['free_so2_ratio'] = (self.df_combined['free sulfur dioxide'] / 
                                              self.df_combined['total sulfur dioxide'])
        print("✓ free_so2_ratio = free sulfur dioxide / total sulfur dioxide")
        
        # 3. Densidade ajustada (removendo efeito do álcool)
        self.df_combined['density_adjusted'] = (self.df_combined['density'] * 
                                                self.df_combined['alcohol'])
        print("✓ density_adjusted = density * alcohol")
        
        # 4. Índice de acidez (pH vs acidez total)
        self.df_combined['acidity_index'] = (self.df_combined['total_acidity'] / 
                                             self.df_combined['pH'])
        print("✓ acidity_index = total_acidity / pH")
        
        # 5. Razão açúcar/álcool (indicador de fermentação)
        self.df_combined['sugar_alcohol_ratio'] = (self.df_combined['residual sugar'] / 
                                                   self.df_combined['alcohol'])
        print("✓ sugar_alcohol_ratio = residual sugar / alcohol")
        
        # 6. Cloretos ajustados por acidez
        self.df_combined['chlorides_adjusted'] = (self.df_combined['chlorides'] * 
                                                  self.df_combined['pH'])
        print("✓ chlorides_adjusted = chlorides * pH")
        
        # 7. Binning de qualidade (Baixa, Média, Alta)
        self.df_combined['quality_class'] = pd.cut(self.df_combined['quality'], 
                                                   bins=[0, 5, 6, 10], 
                                                   labels=['Baixa', 'Média', 'Alta'])
        print("✓ quality_class = Categorização da qualidade (Baixa: ≤5, Média: 6, Alta: ≥7)")
        
        # Atualizar lista de features
        new_features = ['total_acidity', 'free_so2_ratio', 'density_adjusted', 
                       'acidity_index', 'sugar_alcohol_ratio', 'chlorides_adjusted']
        
        print(f"\n✓ Total de novas features criadas: {len(new_features)}")
        print(f"✓ Total de features disponíveis: {len(self.feature_names) + len(new_features)}")
        
        # Estatísticas das novas features
        print("\n--- Estatísticas das Novas Features ---")
        print(self.df_combined[new_features].describe())
        
        # Verificar correlação das novas features com qualidade
        new_features_corr = self.df_combined[new_features + ['quality']].corr()['quality'].drop('quality')
        print("\n--- Correlação das Novas Features com Qualidade ---")
        print(new_features_corr.sort_values(ascending=False))
        
    def train_quality_models(self):
        """
        Treina múltiplos modelos para predição de qualidade
        """
        print("\n" + "="*80)
        print("8. MODELAGEM - PREDIÇÃO DE QUALIDADE")
        print("="*80)
        
        # Preparar dados
        feature_list = (self.feature_names + 
                       ['total_acidity', 'free_so2_ratio', 'density_adjusted', 
                        'acidity_index', 'sugar_alcohol_ratio', 'chlorides_adjusted'])
        
        X = self.df_combined[feature_list]
        y = self.df_combined['quality']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, stratify=y)
        
        # Padronização (usando RobustScaler devido aos outliers)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✓ Dados de treino: {X_train.shape[0]} amostras")
        print(f"✓ Dados de teste: {X_test.shape[0]} amostras")
        print(f"✓ Features utilizadas: {X_train.shape[1]}")
        
        # Definir modelos
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=20, 
                                                   random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, 
                                                   min_samples_split=10, random_state=42, 
                                                   n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                                           learning_rate=0.1, random_state=42),
            'SVR': SVR(kernel='rbf', C=10, gamma='scale')
        }
        
        # Treinar e avaliar modelos
        results = []
        
        print("\n--- Treinando e Avaliando Modelos ---")
        print("-" * 80)
        
        for name, model in models.items():
            print(f"\nTreinando {name}...")
            
            # Treinar
            model.fit(X_train_scaled, y_train)
            
            # Predições
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Métricas
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                       cv=5, scoring='neg_mean_absolute_error', 
                                       n_jobs=-1)
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            results.append({
                'Model': name,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'CV MAE': cv_mae,
                'CV Std': cv_std
            })
            
            # Armazenar modelo e predições
            self.models_quality[name] = {
                'model': model,
                'y_test': y_test,
                'y_pred': y_test_pred,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
            
            print(f"  Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
            print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
            print(f"  CV MAE: {cv_mae:.4f} (±{cv_std:.4f})")
        
        # Resultados consolidados
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test MAE')
        
        print("\n" + "="*80)
        print("RESUMO DOS RESULTADOS - PREDIÇÃO DE QUALIDADE")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Melhor modelo
        best_model_name = results_df.iloc[0]['Model']
        best_mae = results_df.iloc[0]['Test MAE']
        best_r2 = results_df.iloc[0]['Test R²']
        
        print(f"\n🏆 MELHOR MODELO: {best_model_name}")
        print(f"   Test MAE: {best_mae:.4f}")
        print(f"   Test R²: {best_r2:.4f}")
        
        # Visualizações
        self._plot_quality_predictions(results_df)
        
        # Feature importance para Random Forest
        if 'Random Forest' in self.models_quality:
            self._plot_feature_importance()
        
        return results_df
    
    def _plot_quality_predictions(self, results_df):
        """
        Plota predições dos modelos de qualidade
        """
        print("\n--- Gerando Visualizações das Predições ---")
        
        # Gráfico de comparação de métricas
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # MAE
        results_df.plot(x='Model', y=['Train MAE', 'Test MAE'], 
                       kind='bar', ax=axes[0, 0], color=['lightblue', 'darkblue'], 
                       edgecolor='black', alpha=0.8)
        axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold', fontsize=14)
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_xlabel('')
        axes[0, 0].legend(['Train', 'Test'])
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE
        results_df.plot(x='Model', y=['Train RMSE', 'Test RMSE'], 
                       kind='bar', ax=axes[0, 1], color=['lightcoral', 'darkred'], 
                       edgecolor='black', alpha=0.8)
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontweight='bold', fontsize=14)
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_xlabel('')
        axes[0, 1].legend(['Train', 'Test'])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R²
        results_df.plot(x='Model', y=['Train R²', 'Test R²'], 
                       kind='bar', ax=axes[1, 0], color=['lightgreen', 'darkgreen'], 
                       edgecolor='black', alpha=0.8)
        axes[1, 0].set_title('R² Score', fontweight='bold', fontsize=14)
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_xlabel('')
        axes[1, 0].legend(['Train', 'Test'])
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # CV MAE com barras de erro
        axes[1, 1].bar(range(len(results_df)), results_df['CV MAE'], 
                      yerr=results_df['CV Std'], capsize=5, 
                      color='orange', edgecolor='black', alpha=0.8)
        axes[1, 1].set_xticks(range(len(results_df)))
        axes[1, 1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
        axes[1, 1].set_title('Cross-Validation MAE (5-Fold)', fontweight='bold', fontsize=14)
        axes[1, 1].set_ylabel('CV MAE')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/09_quality_model_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 09_quality_model_comparison.png")
        
        # Scatter plots (Real vs Predito) para os 3 melhores modelos
        top_3_models = list(results_df.head(3)['Model'])
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, model_name in enumerate(top_3_models):
            y_test = self.models_quality[model_name]['y_test']
            y_pred = self.models_quality[model_name]['y_pred']
            mae = self.models_quality[model_name]['test_mae']
            r2 = self.models_quality[model_name]['test_r2']
            
            axes[idx].scatter(y_test, y_pred, alpha=0.5, edgecolor='black', 
                            linewidth=0.5, s=50)
            axes[idx].plot([y_test.min(), y_test.max()], 
                          [y_test.min(), y_test.max()], 
                          'r--', lw=2, label='Predição Perfeita')
            axes[idx].set_xlabel('Qualidade Real', fontsize=12)
            axes[idx].set_ylabel('Qualidade Predita', fontsize=12)
            axes[idx].set_title(f'{model_name}\nMAE: {mae:.3f} | R²: {r2:.3f}', 
                              fontweight='bold', fontsize=12)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/10_quality_predictions_scatter.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 10_quality_predictions_scatter.png")
    
    def _plot_feature_importance(self):
        """
        Plota importância das features para Random Forest
        """
        print("\n--- Analisando Feature Importance (Random Forest) ---")
        
        rf_model = self.models_quality['Random Forest']['model']
        
        feature_list = (self.feature_names + 
                       ['total_acidity', 'free_so2_ratio', 'density_adjusted', 
                        'acidity_index', 'sugar_alcohol_ratio', 'chlorides_adjusted'])
        
        feature_importance = pd.DataFrame({
            'Feature': feature_list,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        feature_importance.plot(x='Feature', y='Importance', kind='barh', 
                               ax=ax, color='forestgreen', edgecolor='black', 
                               alpha=0.8, legend=False)
        ax.set_title('Feature Importance - Random Forest\n(Predição de Qualidade)', 
                    fontweight='bold', fontsize=16)
        ax.set_xlabel('Importância', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/11_feature_importance_quality.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 11_feature_importance_quality.png")
    
    def train_type_classification_models(self):
        """
        Treina modelos para classificar tipo de vinho (tinto vs branco)
        baseado apenas em propriedades químicas
        """
        print("\n" + "="*80)
        print("9. MODELAGEM - CLASSIFICAÇÃO DE TIPO (TINTO vs BRANCO)")
        print("="*80)
        print("\n🔬 Objetivo: Verificar se o 'nariz químico' consegue distinguir")
        print("   vinhos tintos de brancos apenas pelas propriedades físico-químicas")
        print("="*80)
        
        # Preparar dados (APENAS features físico-químicas originais)
        X = self.df_combined[self.feature_names]
        y = (self.df_combined['type'] == 'red').astype(int)  # 1 = tinto, 0 = branco
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, stratify=y)
        
        # Padronização
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\n✓ Dados de treino: {X_train.shape[0]} amostras")
        print(f"✓ Dados de teste: {X_test.shape[0]} amostras")
        print(f"✓ Features utilizadas: {X_train.shape[1]} (apenas físico-químicas)")
        
        # Definir modelos de classificação
        models = {
            'Logistic Regression': None,  # Implementar manualmente depois
            'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=20, 
                                                    random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, 
                                                    min_samples_split=10, random_state=42, 
                                                    n_jobs=-1),
            'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        }
        
        # Adicionar Logistic Regression manualmente
        from sklearn.linear_model import LogisticRegression
        models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
        
        # Treinar e avaliar modelos
        results = []
        
        print("\n--- Treinando e Avaliando Modelos de Classificação ---")
        print("-" * 80)
        
        for name, model in models.items():
            print(f"\nTreinando {name}...")
            
            # Treinar
            model.fit(X_train_scaled, y_train)
            
            # Predições
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Métricas
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            # AUC-ROC
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                test_auc = roc_auc_score(y_test, y_test_proba)
            else:
                y_test_proba = y_test_pred
                test_auc = roc_auc_score(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                       cv=5, scoring='accuracy', n_jobs=-1)
            cv_acc = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results.append({
                'Model': name,
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc,
                'Test AUC-ROC': test_auc,
                'CV Accuracy': cv_acc,
                'CV Std': cv_std
            })
            
            # Armazenar modelo e predições
            self.models_type[name] = {
                'model': model,
                'y_test': y_test,
                'y_pred': y_test_pred,
                'y_proba': y_test_proba,
                'test_acc': test_acc,
                'test_auc': test_auc
            }
            
            print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            print(f"  Test AUC: {test_auc:.4f}")
            print(f"  CV Acc: {cv_acc:.4f} (±{cv_std:.4f})")
        
        # Resultados consolidados
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        
        print("\n" + "="*80)
        print("RESUMO DOS RESULTADOS - CLASSIFICAÇÃO DE TIPO")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Melhor modelo
        best_model_name = results_df.iloc[0]['Model']
        best_acc = results_df.iloc[0]['Test Accuracy']
        best_auc = results_df.iloc[0]['Test AUC-ROC']
        
        print(f"\n🏆 MELHOR MODELO: {best_model_name}")
        print(f"   Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"   Test AUC-ROC: {best_auc:.4f}")
        
        print("\n" + "="*80)
        print("🎯 CONCLUSÃO: 'NARIZ QUÍMICO'")
        print("="*80)
        if best_acc > 0.95:
            print(f"✓ EXCELENTE! O algoritmo consegue distinguir vinhos tintos de brancos")
            print(f"  com {best_acc*100:.2f}% de acurácia baseado APENAS nas propriedades químicas!")
            print(f"  Isso comprova que existe uma 'assinatura química' distinta entre os tipos.")
        elif best_acc > 0.85:
            print(f"✓ MUITO BOM! O algoritmo consegue distinguir bem os tipos de vinho")
            print(f"  com {best_acc*100:.2f}% de acurácia. Há padrões químicos claros.")
        elif best_acc > 0.75:
            print(f"⚠ MODERADO. O algoritmo consegue distinguir os tipos com {best_acc*100:.2f}%")
            print(f"  de acurácia, mas há sobreposição nas características químicas.")
        else:
            print(f"⚠ BAIXO. O algoritmo tem dificuldade ({best_acc*100:.2f}%) em distinguir")
            print(f"  os tipos apenas pela química, indicando alta similaridade.")
        
        # Visualizações
        self._plot_type_classification(results_df)
        
        # Confusion Matrix para melhor modelo
        self._plot_confusion_matrix(best_model_name)
        
        # Feature importance para Random Forest
        if 'Random Forest' in self.models_type:
            self._plot_feature_importance_type()
        
        return results_df
    
    def _plot_type_classification(self, results_df):
        """
        Plota resultados da classificação de tipos
        """
        print("\n--- Gerando Visualizações da Classificação ---")
        
        # Gráfico de comparação de métricas
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy
        results_df.plot(x='Model', y=['Train Accuracy', 'Test Accuracy'], 
                       kind='bar', ax=axes[0], color=['lightblue', 'darkblue'], 
                       edgecolor='black', alpha=0.8)
        axes[0].set_title('Accuracy - Classificação de Tipo', fontweight='bold', fontsize=14)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('')
        axes[0].legend(['Train', 'Test'])
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim([0.5, 1.05])
        
        # AUC-ROC
        results_df.plot(x='Model', y='Test AUC-ROC', 
                       kind='bar', ax=axes[1], color='green', 
                       edgecolor='black', alpha=0.8, legend=False)
        axes[1].set_title('AUC-ROC - Classificação de Tipo', fontweight='bold', fontsize=14)
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_xlabel('')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim([0.5, 1.05])
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/12_type_classification_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 12_type_classification_comparison.png")
        
        # ROC Curves
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name in self.models_type.keys():
            y_test = self.models_type[model_name]['y_test']
            y_proba = self.models_type[model_name]['y_proba']
            auc = self.models_type[model_name]['test_auc']
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Classificação de Tipo de Vinho', 
                    fontweight='bold', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/13_roc_curves_type.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 13_roc_curves_type.png")
    
    def _plot_confusion_matrix(self, model_name):
        """
        Plota matriz de confusão para o modelo especificado
        """
        print(f"\n--- Matriz de Confusão: {model_name} ---")
        
        y_test = self.models_type[model_name]['y_test']
        y_pred = self.models_type[model_name]['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                   square=True, linewidths=1, linecolor='black', ax=ax)
        ax.set_xlabel('Predito', fontsize=12)
        ax.set_ylabel('Real', fontsize=12)
        ax.set_title(f'Matriz de Confusão - {model_name}\n(0 = Branco, 1 = Tinto)', 
                    fontweight='bold', fontsize=14)
        ax.set_xticklabels(['Branco (0)', 'Tinto (1)'])
        ax.set_yticklabels(['Branco (0)', 'Tinto (1)'])
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/14_confusion_matrix_type.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 14_confusion_matrix_type.png")
        
        # Classification Report
        print("\nClassification Report:")
        print("-" * 60)
        print(classification_report(y_test, y_pred, 
                                   target_names=['Branco', 'Tinto']))
    
    def _plot_feature_importance_type(self):
        """
        Plota importância das features para classificação de tipo (Random Forest)
        """
        print("\n--- Analisando Feature Importance para Classificação de Tipo ---")
        
        rf_model = self.models_type['Random Forest']['model']
        
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        feature_importance.plot(x='Feature', y='Importance', kind='barh', 
                               ax=ax, color='darkorange', edgecolor='black', 
                               alpha=0.8, legend=False)
        ax.set_title('Feature Importance - Random Forest\n(Classificação de Tipo: Tinto vs Branco)', 
                    fontweight='bold', fontsize=16)
        ax.set_xlabel('Importância', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/15_feature_importance_type.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo: 15_feature_importance_type.png")
        
        print("\n💡 INSIGHTS:")
        print("-" * 60)
        top_3 = feature_importance.head(3)
        print(f"As 3 features mais importantes para distinguir tipos:")
        for idx, row in top_3.iterrows():
            print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")
        
    def wine_segmentation(self):
        """
        Segmentação de vinhos usando as features mais relevantes
        """
        print("\n" + "="*80)
        print("10. SEGMENTAÇÃO DE VINHOS")
        print("="*80)
        
        # Identificar top 3 features para qualidade (do Random Forest)
        if 'Random Forest' in self.models_quality:
            rf_model = self.models_quality['Random Forest']['model']
            feature_list = (self.feature_names + 
                           ['total_acidity', 'free_so2_ratio', 'density_adjusted', 
                            'acidity_index', 'sugar_alcohol_ratio', 'chlorides_adjusted'])
            
            feature_importance = pd.DataFrame({
                'Feature': feature_list,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            top_features = feature_importance.head(3)['Feature'].tolist()
            
            print(f"\n✓ Top 3 features mais importantes para qualidade:")
            for i, feat in enumerate(top_features, 1):
                print(f"  {i}. {feat}")
            
            # Scatter plot 3D usando as 3 features mais importantes
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(16, 6))
            
            # Plot 1: Colorido por qualidade
            ax1 = fig.add_subplot(121, projection='3d')
            scatter1 = ax1.scatter(self.df_combined[top_features[0]], 
                                  self.df_combined[top_features[1]], 
                                  self.df_combined[top_features[2]], 
                                  c=self.df_combined['quality'], 
                                  cmap='RdYlGn', alpha=0.6, s=20, 
                                  edgecolor='black', linewidth=0.3)
            ax1.set_xlabel(top_features[0], fontsize=10)
            ax1.set_ylabel(top_features[1], fontsize=10)
            ax1.set_zlabel(top_features[2], fontsize=10)
            ax1.set_title('Segmentação por Qualidade', fontweight='bold', fontsize=12)
            plt.colorbar(scatter1, ax=ax1, label='Qualidade', shrink=0.5)
            
            # Plot 2: Colorido por tipo
            ax2 = fig.add_subplot(122, projection='3d')
            colors = ['darkred' if t == 'red' else 'gold' 
                     for t in self.df_combined['type']]
            ax2.scatter(self.df_combined[top_features[0]], 
                       self.df_combined[top_features[1]], 
                       self.df_combined[top_features[2]], 
                       c=colors, alpha=0.6, s=20, 
                       edgecolor='black', linewidth=0.3)
            ax2.set_xlabel(top_features[0], fontsize=10)
            ax2.set_ylabel(top_features[1], fontsize=10)
            ax2.set_zlabel(top_features[2], fontsize=10)
            ax2.set_title('Segmentação por Tipo', fontweight='bold', fontsize=12)
            
            # Legenda customizada
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='darkred', edgecolor='black', label='Tinto'),
                              Patch(facecolor='gold', edgecolor='black', label='Branco')]
            ax2.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig('/mnt/user-data/outputs/16_wine_segmentation_3d.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Gráfico salvo: 16_wine_segmentation_3d.png")
            
            # Análise por segmentos (qualidade)
            print("\n--- Análise por Segmentos de Qualidade ---")
            
            # Criar segmentos
            self.df_combined['quality_segment'] = pd.cut(self.df_combined['quality'], 
                                                         bins=[0, 5, 6, 10], 
                                                         labels=['Baixa (≤5)', 
                                                                'Média (6)', 
                                                                'Alta (≥7)'])
            
            segment_analysis = self.df_combined.groupby('quality_segment')[top_features].agg(['mean', 'std'])
            print("\nMédias e Desvios por Segmento:")
            print(segment_analysis)
            
            # Boxplots por segmento
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for idx, feat in enumerate(top_features):
                self.df_combined.boxplot(column=feat, by='quality_segment', ax=axes[idx])
                axes[idx].set_title(f'{feat}', fontweight='bold', fontsize=12)
                axes[idx].set_xlabel('Segmento de Qualidade', fontsize=10)
                axes[idx].set_ylabel(feat, fontsize=10)
                axes[idx].grid(True, alpha=0.3)
            
            plt.suptitle('Distribuição das Top Features por Segmento de Qualidade', 
                        fontweight='bold', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig('/mnt/user-data/outputs/17_segmentation_boxplots.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Gráfico salvo: 17_segmentation_boxplots.png")
    
    def generate_final_report(self):
        """
        Gera relatório final consolidado
        """
        print("\n" + "="*80)
        print("11. RELATÓRIO FINAL CONSOLIDADO")
        print("="*80)
        
        report = []
        
        report.append("=" * 80)
        report.append("RELATÓRIO TÉCNICO - ANÁLISE DO WINE QUALITY DATASET")
        report.append("Vinho Verde (Portugal) - Cortez et al., 2009")
        report.append("=" * 80)
        
        report.append("\n1. DATASET")
        report.append("-" * 80)
        report.append(f"• Total de amostras: {len(self.df_combined)}")
        report.append(f"• Vinho Tinto: {len(self.df_red)} amostras ({len(self.df_red)/len(self.df_combined)*100:.1f}%)")
        report.append(f"• Vinho Branco: {len(self.df_white)} amostras ({len(self.df_white)/len(self.df_combined)*100:.1f}%)")
        report.append(f"• Features físico-químicas: {len(self.feature_names)}")
        report.append(f"• Range de qualidade: {self.df_combined['quality'].min()} a {self.df_combined['quality'].max()}")
        report.append(f"• Qualidade média (Tinto): {self.df_red['quality'].mean():.2f} ± {self.df_red['quality'].std():.2f}")
        report.append(f"• Qualidade média (Branco): {self.df_white['quality'].mean():.2f} ± {self.df_white['quality'].std():.2f}")
        
        report.append("\n2. PRINCIPAIS DESCOBERTAS - EDA")
        report.append("-" * 80)
        
        # Correlações com qualidade
        quality_corr = self.df_combined[self.feature_names + ['quality']].corr()['quality'].drop('quality')
        top_pos = quality_corr.nlargest(3)
        top_neg = quality_corr.nsmallest(3)
        
        report.append("\nCorrelações Positivas mais Fortes com Qualidade:")
        for feat, corr in top_pos.items():
            report.append(f"  • {feat}: {corr:.3f}")
        
        report.append("\nCorrelações Negativas mais Fortes com Qualidade:")
        for feat, corr in top_neg.items():
            report.append(f"  • {feat}: {corr:.3f}")
        
        report.append("\n3. MODELAGEM - PREDIÇÃO DE QUALIDADE")
        report.append("-" * 80)
        
        if self.models_quality:
            best_model = min(self.models_quality.items(), 
                           key=lambda x: x[1]['test_mae'])
            report.append(f"\n🏆 Melhor Modelo: {best_model[0]}")
            report.append(f"   • Test MAE: {best_model[1]['test_mae']:.4f}")
            report.append(f"   • Test R²: {best_model[1]['test_r2']:.4f}")
            
            report.append("\nRanking de Modelos (por MAE):")
            for rank, (name, results) in enumerate(sorted(self.models_quality.items(), 
                                                         key=lambda x: x[1]['test_mae']), 1):
                report.append(f"  {rank}. {name}: MAE = {results['test_mae']:.4f}, R² = {results['test_r2']:.4f}")
        
        report.append("\n4. MODELAGEM - CLASSIFICAÇÃO DE TIPO (TINTO vs BRANCO)")
        report.append("-" * 80)
        
        if self.models_type:
            best_model = max(self.models_type.items(), 
                           key=lambda x: x[1]['test_acc'])
            report.append(f"\n🏆 Melhor Modelo: {best_model[0]}")
            report.append(f"   • Test Accuracy: {best_model[1]['test_acc']:.4f} ({best_model[1]['test_acc']*100:.2f}%)")
            report.append(f"   • Test AUC-ROC: {best_model[1]['test_auc']:.4f}")
            
            report.append("\nRanking de Modelos (por Accuracy):")
            for rank, (name, results) in enumerate(sorted(self.models_type.items(), 
                                                         key=lambda x: x[1]['test_acc'], 
                                                         reverse=True), 1):
                report.append(f"  {rank}. {name}: Acc = {results['test_acc']:.4f}, AUC = {results['test_auc']:.4f}")
            
            report.append("\n📊 CONCLUSÃO - 'NARIZ QUÍMICO':")
            if best_model[1]['test_acc'] > 0.95:
                report.append("✓ O algoritmo consegue distinguir vinhos tintos de brancos com")
                report.append(f"  EXCELENTE precisão ({best_model[1]['test_acc']*100:.2f}%) baseado APENAS nas")
                report.append("  propriedades físico-químicas, comprovando que existe uma")
                report.append("  'assinatura química' distinta entre os tipos.")
            elif best_model[1]['test_acc'] > 0.85:
                report.append("✓ O algoritmo consegue distinguir bem os tipos de vinho")
                report.append(f"  ({best_model[1]['test_acc']*100:.2f}% de precisão), indicando padrões químicos claros.")
        
        report.append("\n5. FEATURE IMPORTANCE")
        report.append("-" * 80)
        
        if 'Random Forest' in self.models_quality:
            rf_model = self.models_quality['Random Forest']['model']
            feature_list = (self.feature_names + 
                           ['total_acidity', 'free_so2_ratio', 'density_adjusted', 
                            'acidity_index', 'sugar_alcohol_ratio', 'chlorides_adjusted'])
            
            feature_importance = pd.DataFrame({
                'Feature': feature_list,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            report.append("\nTop 5 Features para Predição de Qualidade:")
            for idx, row in feature_importance.iterrows():
                report.append(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        if 'Random Forest' in self.models_type:
            rf_model = self.models_type['Random Forest']['model']
            
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            report.append("\nTop 5 Features para Classificação de Tipo:")
            for idx, row in feature_importance.iterrows():
                report.append(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        report.append("\n6. RECOMENDAÇÕES TÉCNICAS")
        report.append("-" * 80)
        report.append("\n6.1 Para Predição de Qualidade:")
        report.append("  • Algoritmos ensemble (Random Forest, Gradient Boosting) apresentam")
        report.append("    melhor desempenho que modelos lineares")
        report.append("  • Features relacionadas ao álcool, acidez volátil e sulfatos são")
        report.append("    os principais preditores de qualidade")
        report.append("  • Feature engineering (criação de índices e razões) pode melhorar")
        report.append("    marginalmente a performance")
        
        report.append("\n6.2 Para Produção:")
        report.append("  • Aumentar teor alcoólico (dentro dos limites legais) tende a")
        report.append("    melhorar a percepção de qualidade")
        report.append("  • Controlar rigorosamente acidez volátil (ácido acético)")
        report.append("  • Monitorar níveis de sulfatos para otimizar conservação")
        
        report.append("\n6.3 Para Certificação:")
        report.append("  • O modelo de ML pode ser usado como sistema de apoio à decisão")
        report.append("  • Tolerância de ±0.5 pontos é razoável para validação")
        report.append("  • Casos com grande discrepância devem passar por revisão manual")
        
        report.append("\n" + "=" * 80)
        report.append("FIM DO RELATÓRIO")
        report.append("=" * 80)
        
        # Salvar relatório
        report_text = "\n".join(report)
        with open('/mnt/user-data/outputs/RELATORIO_FINAL.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n✓ Relatório salvo: RELATORIO_FINAL.txt")
        
        return report_text


def main():
    """
    Função principal
    """
    print("\n" + "="*80)
    print("WINE QUALITY ANALYSIS - VINHO VERDE DATASET")
    print("Análise Completa: EDA + Machine Learning + Classificação de Tipos")
    print("="*80)
    
    # Paths dos datasets
    red_wine_path = '/mnt/user-data/uploads/winequality-red.csv'
    white_wine_path = '/mnt/user-data/uploads/winequality-white.csv'
    
    # Inicializar analisador
    analyzer = WineQualityAnalyzer(red_wine_path, white_wine_path)
    
    # Pipeline completo
    try:
        # 1. Carregar e combinar dados
        analyzer.load_and_combine_data()
        
        # 2. Análise preliminar
        analyzer.preliminary_analysis()
        
        # 3. EDA
        analyzer.exploratory_data_analysis()
        
        # 4. Análise comparativa
        analyzer.comparative_analysis()
        
        # 5. Análise de correlação
        analyzer.correlation_analysis()
        
        # 6. PCA
        analyzer.pca_analysis()
        
        # 7. Feature Engineering
        analyzer.feature_engineering()
        
        # 8. Modelagem - Qualidade
        analyzer.train_quality_models()
        
        # 9. Modelagem - Tipo (Tinto vs Branco)
        analyzer.train_type_classification_models()
        
        # 10. Segmentação
        analyzer.wine_segmentation()
        
        # 11. Relatório Final
        analyzer.generate_final_report()
        
        print("\n" + "="*80)
        print("✓✓✓ ANÁLISE CONCLUÍDA COM SUCESSO! ✓✓✓")
        print("="*80)
        print(f"\nTodos os resultados foram salvos em: /mnt/user-data/outputs/")
        print("\nArquivos gerados:")
        print("  • 17 gráficos de alta qualidade (PNG 300 DPI)")
        print("  • 1 relatório técnico consolidado (TXT)")
        print("\n🍷 Análise completa do Vinho Verde finalizada!")
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
