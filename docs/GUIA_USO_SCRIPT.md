# 🐍 GUIA DE USO - SCRIPT PYTHON

## 📄 **wine_quality_analysis.py**

**Tamanho**: 68 KB  
**Linhas de Código**: ~1.800 linhas  
**Linguagem**: Python 3.x

---

## 🎯 **O QUE O SCRIPT FAZ**

Este script realiza uma análise **COMPLETA** do Wine Quality Dataset:

✅ **1. Carregamento e Combinação dos Dados**
- Carrega vinhos tintos e brancos
- Adiciona coluna 'type' (red/white)
- Remove duplicatas automaticamente

✅ **2. Análise Preliminar**
- Info do dataset
- Estatísticas descritivas
- Verificação de missing values e duplicatas

✅ **3. EDA (Análise Exploratória)**
- Distribuições de todas as features
- Análise de outliers (método IQR)
- Testes de normalidade (Shapiro-Wilk, D'Agostino)

✅ **4. Análise Comparativa (Tinto vs Branco)**
- Estatísticas comparativas
- Violin plots com testes estatísticos (Mann-Whitney U)
- Análise de qualidade por tipo

✅ **5. Análise de Correlação**
- Matriz de correlação completa
- Correlações com qualidade
- Detecção de multicolinearidade

✅ **6. Análise PCA**
- Variância explicada
- Scree plot
- Biplot 2D
- Loadings das features

✅ **7. Feature Engineering**
- Criação de 6 novas features:
  - total_acidity
  - free_so2_ratio
  - density_adjusted
  - acidity_index
  - sugar_alcohol_ratio
  - chlorides_adjusted

✅ **8. Modelagem - Predição de Qualidade**
- 7 modelos testados:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - SVR
- Cross-validation 5-fold
- Feature importance (Random Forest)

✅ **9. Modelagem - Classificação de Tipo**
- 4 modelos testados:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
- Matriz de confusão
- Curvas ROC
- Feature importance

✅ **10. Segmentação de Vinhos**
- Scatter 3D com top features
- Análise por segmentos de qualidade

✅ **11. Relatório Final**
- Consolidação de todas as descobertas

---

## 🚀 **COMO EXECUTAR**

### **Opção 1: Executar Diretamente**

```bash
python wine_quality_analysis.py
```

### **Opção 2: Executar no Jupyter/Colab**

```python
# Execute célula por célula para análise interativa
%run wine_quality_analysis.py
```

### **Opção 3: Importar como Módulo**

```python
from wine_quality_analysis import WineQualityAnalyzer

# Inicializar
analyzer = WineQualityAnalyzer(
    red_wine_path='winequality-red.csv',
    white_wine_path='winequality-white.csv'
)

# Executar análise completa
analyzer.load_and_combine_data()
analyzer.preliminary_analysis()
analyzer.exploratory_data_analysis()
analyzer.comparative_analysis()
analyzer.correlation_analysis()
analyzer.pca_analysis()
analyzer.feature_engineering()
analyzer.train_quality_models()
analyzer.train_type_classification_models()
analyzer.wine_segmentation()
analyzer.generate_final_report()
```

---

## 📦 **DEPENDÊNCIAS**

### **Bibliotecas Necessárias**

```python
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, roc_curve)
```

### **Instalação via pip**

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### **Instalação via conda**

```bash
conda install numpy pandas matplotlib seaborn scipy scikit-learn
```

---

## ⚙️ **CONFIGURAÇÕES**

### **Padrões do Script**

```python
# Visualizações
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Padronização
scaler = RobustScaler()  # Robusto a outliers

# Validação
test_size = 0.2  # 80% treino, 20% teste
cv_folds = 5     # 5-fold cross-validation

# Random State
random_state = 42  # Reprodutibilidade
```

### **Como Customizar**

```python
# Exemplo: Mudar estilo dos gráficos
plt.style.use('ggplot')  # ou 'seaborn', 'fivethirtyeight', etc.

# Exemplo: Ajustar tamanho das figuras
plt.rcParams['figure.figsize'] = (16, 8)

# Exemplo: Usar StandardScaler ao invés de RobustScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```

---

## 📊 **OUTPUTS GERADOS**

O script salva automaticamente:

- **17 gráficos PNG** (300 DPI) em `/mnt/user-data/outputs/`
- **1 relatório TXT** consolidado

---

## ⏱️ **TEMPO DE EXECUÇÃO**

| Etapa | Tempo Aprox. |
|-------|-------------|
| Carregamento de dados | < 1 segundo |
| EDA | 10-15 segundos |
| Análise PCA | 2-3 segundos |
| Modelagem Qualidade | 60-90 segundos |
| Modelagem Tipo | 30-45 segundos |
| Segmentação | 5-10 segundos |
| **TOTAL** | **~3-5 minutos** |

*Nota: Tempo pode variar conforme hardware*

---

## 🎨 **ESTRUTURA DO CÓDIGO**

### **Classe Principal: `WineQualityAnalyzer`**

```python
class WineQualityAnalyzer:
    def __init__(self, red_wine_path, white_wine_path)
    def load_and_combine_data(self)
    def preliminary_analysis(self)
    def exploratory_data_analysis(self)
    def comparative_analysis(self)
    def correlation_analysis(self)
    def pca_analysis(self)
    def feature_engineering(self)
    def train_quality_models(self)
    def train_type_classification_models(self)
    def wine_segmentation(self)
    def generate_final_report(self)
    
    # Métodos auxiliares privados
    def _plot_distributions(self)
    def _analyze_outliers(self)
    def _test_normality(self)
    def _plot_quality_predictions(self, results_df)
    def _plot_feature_importance(self)
    def _plot_type_classification(self, results_df)
    def _plot_confusion_matrix(self, model_name)
    def _plot_feature_importance_type(self)
```

---

## 🔧 **PERSONALIZAÇÃO AVANÇADA**

### **Exemplo 1: Adicionar Novo Modelo**

```python
# No método train_quality_models(), adicione:
from sklearn.ensemble import AdaBoostRegressor

models = {
    # ... modelos existentes ...
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42)
}
```

### **Exemplo 2: Mudar Métrica de Avaliação**

```python
# No lugar de MAE, usar MSE:
from sklearn.metrics import mean_squared_error

# Substituir:
test_mae = mean_absolute_error(y_test, y_test_pred)

# Por:
test_mse = mean_squared_error(y_test, y_test_pred)
```

### **Exemplo 3: Ajustar Hiperparâmetros**

```python
# Random Forest com mais árvores:
'Random Forest': RandomForestRegressor(
    n_estimators=200,      # Era 100
    max_depth=20,          # Era 15
    min_samples_split=5,   # Era 10
    random_state=42,
    n_jobs=-1
)
```

---

## 🐛 **TROUBLESHOOTING**

### **Erro: ModuleNotFoundError**

```bash
# Solução: Instalar biblioteca faltante
pip install nome-da-biblioteca
```

### **Erro: FileNotFoundError**

```python
# Solução: Verificar caminhos dos arquivos CSV
red_wine_path = 'caminho/correto/winequality-red.csv'
white_wine_path = 'caminho/correto/winequality-white.csv'
```

### **Aviso: ConvergenceWarning (Lasso/Ridge)**

```python
# Solução: Aumentar max_iter
from sklearn.linear_model import Lasso, Ridge

Lasso(alpha=0.1, max_iter=10000)  # Aumentar de 1000 para 10000
```

### **Performance Lenta**

```python
# Solução 1: Reduzir cross-validation folds
cv_folds = 3  # Era 5

# Solução 2: Reduzir n_estimators
'Random Forest': RandomForestRegressor(n_estimators=50)  # Era 100
```

---

## 📚 **DOCUMENTAÇÃO DO CÓDIGO**

### **Docstrings Completas**

Todas as funções têm docstrings detalhadas:

```python
def load_and_combine_data(self):
    """
    Carrega e combina os datasets de vinho tinto e branco
    
    Returns:
    --------
    pd.DataFrame
        Dataset combinado com coluna 'type' adicionada
    """
```

### **Comentários Inline**

```python
# Adicionar coluna de tipo
self.df_red['type'] = 'red'
self.df_white['type'] = 'white'

# Combinar datasets
self.df_combined = pd.concat([self.df_red, self.df_white], 
                              axis=0, ignore_index=True)
```

---

## 🎓 **CONCEITOS APLICADOS**

### **Machine Learning**
- Regressão (Linear, Ridge, Lasso, Tree, Forest, Boosting, SVR)
- Classificação (Logistic, Tree, Forest, SVM)
- Cross-validation
- Hyperparameter tuning
- Feature engineering
- Feature importance

### **Estatística**
- Testes de normalidade
- Teste de Mann-Whitney U
- Correlação de Pearson
- Análise de outliers (IQR)
- Intervalos de confiança

### **Visualização**
- Histogramas
- Boxplots
- Violin plots
- Heatmaps
- Scatter plots
- ROC curves
- Confusion matrix

---

## 🔐 **BOAS PRÁTICAS IMPLEMENTADAS**

✅ **Código Limpo**
- Nomes descritivos
- Funções modulares
- Comentários adequados

✅ **Reprodutibilidade**
- Random state fixo (42)
- Versões de bibliotecas documentadas

✅ **Escalabilidade**
- Classe orientada a objetos
- Métodos reutilizáveis

✅ **Validação**
- Cross-validation
- Holdout test set
- Múltiplas métricas

✅ **Visualizações**
- Alta resolução (300 DPI)
- Cores consistentes
- Títulos informativos

---

## 🚀 **USO EM PRODUÇÃO**

### **Exemplo: API Flask**

```python
from flask import Flask, request, jsonify
from wine_quality_analysis import WineQualityAnalyzer
import pickle

app = Flask(__name__)

# Carregar modelo treinado
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'quality': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
```

### **Exemplo: Jupyter Notebook Interativo**

```python
# Célula 1: Import
from wine_quality_analysis import WineQualityAnalyzer

# Célula 2: Inicializar
analyzer = WineQualityAnalyzer('red.csv', 'white.csv')

# Célula 3: Carregar dados
df = analyzer.load_and_combine_data()

# Célula 4: Análise interativa
analyzer.exploratory_data_analysis()

# ... e assim por diante
```

---

## 📧 **SUPORTE**

**Dúvidas sobre o código?**
- Leia os docstrings das funções
- Consulte os comentários inline
- Veja os exemplos neste guia

**Erro não documentado?**
- Verifique versões das bibliotecas
- Teste com dataset de exemplo
- Consulte documentação do scikit-learn

---

## 🍷 **CITAÇÃO**

```bibtex
@software{wine_quality_analysis,
  title = {Wine Quality Analysis - Complete EDA and ML Pipeline},
  author = {Análise para Lineu},
  year = {2025},
  note = {Dataset: Cortez et al., 2009}
}
```

---

**🎯 FIM DO GUIA DE USO DO SCRIPT** ✨
