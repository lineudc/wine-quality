# 🐍 GUIA DE USO - WINE QUALITY ANALYSIS

## � **Estrutura do Projeto**

O projeto foi modernizado e dividido em módulos para facilitar a manutenção e escalabilidade.

```
wine-quality/
├── data/                     # Dados brutos e processados
├── docs/                     # Documentação
├── outputs/                  # Gráficos e relatórios gerados
├── src/                      # Código fonte
│   ├── config.py             # Configurações
│   ├── data_loader.py        # Carregamento de dados
│   ├── eda.py                # Análise Exploratória
│   ├── features.py           # Engenharia de Features
│   ├── models.py             # Modelos de ML
│   ├── visualization.py      # Visualizações
│   └── main.py               # Script principal
├── tests/                    # Testes unitários
├── Makefile                  # Automação
└── requirements.txt          # Dependências
```

---

## 🚀 **COMO EXECUTAR**

Utilizamos um `Makefile` para simplificar os comandos.

### **1. Configuração Inicial (Setup)**

Cria o ambiente virtual (`.venv`) e instala as dependências automaticamente:

```bash
make setup
```

### **2. Executar Análise**

Roda todo o pipeline de análise (carregamento, EDA, ML, relatórios):

```bash
make run
```

### **3. Rodar Testes**

Executa os testes unitários para garantir que tudo está funcionando:

```bash
make test
```

### **4. Limpeza**

Remove arquivos temporários e caches:

```bash
make clean
```

---

## 📦 **DEPENDÊNCIAS**

As dependências estão listadas em `requirements.txt`.

### **Principais Bibliotecas**
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **matplotlib & seaborn**: Visualização de dados
- **scikit-learn**: Machine Learning
- **scipy**: Testes estatísticos

### **Instalação Manual (sem Makefile)**

Caso prefira não usar o Makefile:

```bash
# Criar venv
python3 -m venv .venv

# Ativar venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

---

## ⚙️ **CONFIGURAÇÕES**

As configurações globais estão em `src/config.py`.

### **Caminhos e Parâmetros**

```python
# src/config.py

# Caminhos dos dados
RED_WINE_PATH = RAW_DATA_DIR / "winequality-red.csv"
WHITE_WINE_PATH = RAW_DATA_DIR / "winequality-white.csv"

# Configurações de Plotagem
PLOT_SETTINGS = {
    'style': 'seaborn-v0_8-darkgrid',
    'figsize': (12, 6),
    ...
}
```

Para alterar cores, tamanhos de gráficos ou caminhos de arquivos, edite este arquivo.

---

## 📊 **OUTPUTS GERADOS**

Ao rodar `make run`, os resultados são salvos em `outputs/`:

- **01_feature_distributions.png**: Distribuição de todas as variáveis.
- **02_quality_distribution.png**: Qualidade por tipo de vinho.
- **03_outliers_boxplots.png**: Análise de outliers.
- **04_red_vs_white_comparison.png**: Comparação visual entre tintos e brancos.
- **05_correlation_matrix.png**: Matriz de correlação completa.
- **06_quality_correlations.png**: Correlações específicas com a qualidade.
- **07_pca_variance.png**: Variância explicada pelo PCA.
- **08_pca_biplot.png**: Visualização 2D dos componentes principais.

---

## � **TROUBLESHOOTING**

### **Erro: `make: command not found`**
Se você não tem o `make` instalado (comum no Windows), use os comandos manuais listados na seção "Instalação Manual".

### **Erro: `ModuleNotFoundError`**
Certifique-se de que ativou o ambiente virtual ou usou `make run` (que usa o python do ambiente virtual automaticamente).

### **Erro ao carregar dados**
Verifique se os arquivos `winequality-red.csv` e `winequality-white.csv` estão na pasta `data/raw/`.

---

**🎯 FIM DO GUIA**
