# 📂 GUIA DE ARQUIVOS - ANÁLISE WINE QUALITY

## 🎯 **ÍNDICE GERAL**

Foram gerados **20 arquivos** no total:
- **17 gráficos** (PNG, 300 DPI, alta qualidade)
- **3 documentos** de texto/análise

---

## 📊 **GRÁFICOS (PNG - 300 DPI)**

### **🔍 Análise Exploratória (EDA)**

#### 1️⃣ `01_feature_distributions.png` (746 KB)
**O que mostra**: Histogramas de todas as 11 features físico-químicas  
**Para quê**: Entender distribuições, detectar assimetrias e outliers  
**Insight chave**: Maioria das features não segue distribuição normal

#### 2️⃣ `02_quality_distribution.png` (141 KB)
**O que mostra**: Distribuição da qualidade (geral, tintos, brancos)  
**Para quê**: Ver concentração de notas (5-6 são maioria)  
**Insight chave**: Distribuição assimétrica, poucos vinhos 8-9

#### 3️⃣ `03_outliers_boxplots.png` (559 KB)
**O que mostra**: Boxplots para detecção de outliers em cada feature  
**Para quê**: Identificar valores extremos  
**Insight chave**: Açúcar residual tem mais outliers (10.3%)

---

### **🔴⚪ Comparação Tintos vs Brancos**

#### 4️⃣ `04_red_vs_white_comparison.png` (737 KB)
**O que mostra**: Violin plots comparando todas as features entre tipos  
**Para quê**: Ver diferenças estatísticas (com p-values)  
**Insight chave**: SO₂ Total tem diferença de +200% (brancos >> tintos)

---

### **📈 Correlações**

#### 5️⃣ `05_correlation_matrix.png` (396 KB)
**O que mostra**: Heatmap completo de correlações (11 features + qualidade)  
**Para quê**: Identificar relações lineares e multicolinearidade  
**Insight chave**: Álcool vs qualidade (+0.47), Densidade vs álcool (-0.78)

#### 6️⃣ `06_quality_correlations.png` (260 KB)
**O que mostra**: Barras horizontais das correlações com qualidade (geral, tinto, branco)  
**Para quê**: Comparar o que importa para cada tipo  
**Insight chave**: Álcool é o campeão para ambos os tipos

---

### **🧬 Análise PCA**

#### 7️⃣ `07_pca_variance_explained.png` (274 KB)
**O que mostra**: Scree plot + curva de variância acumulada  
**Para quê**: Determinar quantos componentes são necessários  
**Insight chave**: 8 componentes para 95% da variância (não trivialmente redutível)

#### 8️⃣ `08_pca_biplot.png` (2.4 MB)
**O que mostra**: Scatter PC1 vs PC2 (colorido por qualidade e tipo)  
**Para quê**: Visualizar separação/agrupamento  
**Insight chave**: Não há separação clara em apenas 2 dimensões

---

### **🤖 Modelagem - Predição de Qualidade**

#### 9️⃣ `09_quality_model_comparison.png` (427 KB)
**O que mostra**: Comparação de 7 modelos (MAE, RMSE, R², CV MAE)  
**Para quê**: Escolher o melhor modelo  
**Insight chave**: Random Forest vence com MAE = 0.54

#### 🔟 `10_quality_predictions_scatter.png` (728 KB)
**O que mostra**: Real vs Predito para os 3 melhores modelos  
**Para quê**: Avaliar visualmente a qualidade das predições  
**Insight chave**: Predições concentradas, mas com dispersão moderada

#### 1️⃣1️⃣ `11_feature_importance_quality.png` (215 KB)
**O que mostra**: Importância das features no Random Forest  
**Para quê**: Saber o que mais influencia a qualidade  
**Insight chave**: Density Adjusted (30.1%) é o líder

---

### **🔴⚪ Modelagem - Classificação de Tipo**

#### 1️⃣2️⃣ `12_type_classification_comparison.png` (161 KB)
**O que mostra**: Comparação de 4 modelos (Accuracy, AUC-ROC)  
**Para quê**: Escolher o melhor classificador  
**Insight chave**: SVM vence com 99.53% de accuracy! 🚀

#### 1️⃣3️⃣ `13_roc_curves_type.png` (232 KB)
**O que mostra**: Curvas ROC para todos os 4 modelos  
**Para quê**: Avaliar trade-off sensibilidade/especificidade  
**Insight chave**: Todos os modelos têm AUC > 0.96 (excelente)

#### 1️⃣4️⃣ `14_confusion_matrix_type.png` (96 KB)
**O que mostra**: Matriz de confusão do SVM  
**Para quê**: Ver erros específicos (falsos positivos/negativos)  
**Insight chave**: Apenas 5 erros em 1.064 amostras!

#### 1️⃣5️⃣ `15_feature_importance_type.png` (166 KB)
**O que mostra**: Importância das features para classificar tipos  
**Para quê**: Entender a "assinatura química" de cada tipo  
**Insight chave**: SO₂ Total (31.5%) e Cloretos (24.2%) são determinantes

---

### **🎯 Segmentação**

#### 1️⃣6️⃣ `16_wine_segmentation_3d.png` (2.0 MB)
**O que mostra**: Scatter 3D com as 3 features mais importantes  
**Para quê**: Visualizar clusters e padrões complexos  
**Insight chave**: Separação clara por tipo, gradiente contínuo por qualidade

#### 1️⃣7️⃣ `17_segmentation_boxplots.png` (296 KB)
**O que mostra**: Boxplots das top features por segmento de qualidade (Baixa/Média/Alta)  
**Para quê**: Comparar perfis químicos de diferentes níveis  
**Insight chave**: Vinhos de alta qualidade têm densidade ajustada mais alta

---

## 📄 **DOCUMENTOS DE TEXTO**

### 📋 `RELATORIO_FINAL.txt` (4 KB)
**Formato**: Texto puro (TXT)  
**Conteúdo**: Relatório técnico consolidado com todas as estatísticas  
**Para quê**: Referência rápida, copiar para apresentações  
**Leitura**: ~2-3 minutos

### 📖 `ANALISE_COMPLETA_WINE_QUALITY.md` (Novo!)
**Formato**: Markdown  
**Conteúdo**: Documento COMPLETO com:
- Resumo executivo
- Todas as descobertas detalhadas
- Tabelas formatadas
- Interpretações enológicas
- Recomendações práticas
- Referências
**Para quê**: Documentação definitiva, apresentações técnicas  
**Leitura**: ~15-20 minutos

### 🎨 `INFOGRAFICO_VISUAL.txt` (Novo!)
**Formato**: ASCII Art / Texto formatado  
**Conteúdo**: Infográfico visual com:
- Principais números
- Gráficos em ASCII
- Comparações visuais
- Conclusões destacadas
**Para quê**: Apresentações, resumo visual rápido  
**Leitura**: ~5 minutos

---

## 🗂️ **COMO USAR OS ARQUIVOS**

### **Para Apresentação Executiva**
1. Comece com: `INFOGRAFICO_VISUAL.txt`
2. Mostre: `12_type_classification_comparison.png` (resultado do "nariz químico")
3. Mostre: `09_quality_model_comparison.png` (modelos de predição)
4. Finalize com recomendações do `ANALISE_COMPLETA_WINE_QUALITY.md`

### **Para Análise Técnica Profunda**
1. Leia: `ANALISE_COMPLETA_WINE_QUALITY.md` (documento mestre)
2. Consulte gráficos na ordem (01 a 17)
3. Use: `RELATORIO_FINAL.txt` para copiar estatísticas

### **Para Publicação Científica**
1. Base: `ANALISE_COMPLETA_WINE_QUALITY.md`
2. Figuras: Todos os gráficos (já em 300 DPI)
3. Metodologia: Seção técnica do documento MD
4. Referências: Incluídas no documento MD

### **Para Implementação Prática**
1. Foco: Seção "Recomendações Técnicas" (documento MD)
2. Features críticas: `11_feature_importance_quality.png`
3. Workflow: Diagrama de certificação (documento MD)

---

## 📊 **ESTATÍSTICAS DOS ARQUIVOS**

| Tipo | Quantidade | Tamanho Total |
|------|-----------|---------------|
| Gráficos PNG | 17 | ~9.7 MB |
| Documentos TXT/MD | 3 | ~50 KB |
| **TOTAL** | **20** | **~9.75 MB** |

---

## 🔍 **GRÁFICOS POR TAMANHO**

**Maiores** (> 1 MB):
- `08_pca_biplot.png` (2.4 MB)
- `16_wine_segmentation_3d.png` (2.0 MB)

**Médios** (200-800 KB):
- `01_feature_distributions.png` (746 KB)
- `04_red_vs_white_comparison.png` (737 KB)
- `10_quality_predictions_scatter.png` (728 KB)

**Menores** (< 200 KB):
- `14_confusion_matrix_type.png` (96 KB)
- `02_quality_distribution.png` (141 KB)

---

## 🎯 **FLUXO DE LEITURA SUGERIDO**

### **Iniciante em Vinho**
1. `INFOGRAFICO_VISUAL.txt` → Visão geral
2. `02_quality_distribution.png` → Entender as notas
3. `06_quality_correlations.png` → O que importa
4. `12_type_classification_comparison.png` → Tinto vs Branco

### **Enólogo / Produtor**
1. `ANALISE_COMPLETA_WINE_QUALITY.md` → Documento completo
2. `11_feature_importance_quality.png` → Focar no importante
3. `17_segmentation_boxplots.png` → Perfis de qualidade
4. Seção "Recomendações Práticas" (MD)

### **Cientista de Dados**
1. `RELATORIO_FINAL.txt` → Estatísticas técnicas
2. `09_quality_model_comparison.png` → Performance dos modelos
3. `05_correlation_matrix.png` → Multicolinearidade
4. `07_pca_variance_explained.png` → Redução de dimensionalidade

### **Gerente de Certificação**
1. Seção "Sistema de Apoio à Decisão" (MD)
2. `10_quality_predictions_scatter.png` → Precisão das predições
3. `14_confusion_matrix_type.png` → Erros esperados
4. Workflow de implementação (MD)

---

## 📌 **PRINCIPAIS DESCOBERTAS (RESUMO RÁPIDO)**

### **1. Predição de Qualidade**
- ✅ Viável com Random Forest (MAE = 0.54)
- 🔑 Álcool é o preditor #1
- ⚠️ Subjetividade limita R² a ~0.37

### **2. Classificação de Tipo**
- ✅✅✅ EXCELENTE com SVM (99.53% accuracy)
- 🔑 SO₂ Total distingue claramente
- 🎯 "Assinatura química" comprovada!

### **3. Features Críticas**
- 🏆 **Qualidade**: Álcool (+), Acidez Volátil (-)
- 🏆 **Tipo**: SO₂ Total, Cloretos, Acidez Volátil

### **4. Aplicações Práticas**
- 🏭 Controle de processo
- 📋 Sistema de certificação
- 📊 Segmentação de mercado
- 🔬 P&D de novos produtos

---

## 🛠️ **FERRAMENTAS USADAS**

- **Python 3.x**
- **Bibliotecas**: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
- **Modelos**: 7 regressão + 4 classificação = 11 total
- **Validação**: 5-fold cross-validation + holdout 80/20
- **Tempo**: ~3-5 minutos de execução

---

## 📧 **CONTATO E CITAÇÃO**

**Análise elaborada para**: Lineu  
**Data**: Novembro 2025  
**Dataset Original**: Cortez et al., 2009  

**Como citar**:
```
Análise Wine Quality Dataset (Vinho Verde)
Dataset original: Cortez et al., 2009
Análise completa com EDA + ML + Classificação de Tipos
Novembro 2025
```

---

## 🍷 **FRASE FINAL**

*"In vino veritas, in data sapientia"*  
*(No vinho está a verdade, nos dados está a sabedoria)*

---

**🎯 FIM DO GUIA** ✨
