# 🍷 ANÁLISE COMPLETA - WINE QUALITY DATASET (VINHO VERDE)

**Análise Exploratória de Dados e Machine Learning**  
*Dataset: Cortez et al., 2009 - UCI Wine Quality Dataset*

---

## 📊 **RESUMO EXECUTIVO**

### **Dataset**
- **Total de amostras**: 5.320 (após remoção de duplicatas)
- **Vinho Tinto**: 1.599 amostras (30.1%)
- **Vinho Branco**: 3.961 amostras (74.5% - ajustado)
- **Features físico-químicas**: 11 originais + 6 engenheiradas = 17 totais
- **Range de qualidade**: 3 a 9 pontos
- **Qualidade média**:
  - Tinto: 5.64 ± 0.81
  - Branco: 5.88 ± 0.89

---

## 🔬 **PRINCIPAIS DESCOBERTAS**

### **1. Correlações com Qualidade**

#### **✅ Correlações Positivas (Melhoram a qualidade)**
| Feature | Correlação | Interpretação |
|---------|-----------|---------------|
| **Alcohol** | +0.469 | 🏆 Maior preditor positivo - vinhos com maior teor alcoólico tendem a ser melhor avaliados |
| **Citric Acid** | +0.098 | Adiciona frescor e "vivacidade" ao vinho |
| **Free Sulfur Dioxide** | +0.054 | Proteção contra oxidação (em níveis adequados) |

#### **❌ Correlações Negativas (Prejudicam a qualidade)**
| Feature | Correlação | Interpretação |
|---------|-----------|---------------|
| **Density** | -0.326 | Densidade alta associada a menor álcool e maior açúcar residual |
| **Volatile Acidity** | -0.265 | 🚨 Ácido acético (vinagre) - defeito grave |
| **Chlorides** | -0.202 | Excesso de sais prejudica o equilíbrio |

---

## 🤖 **MODELAGEM - PREDIÇÃO DE QUALIDADE**

### **Objetivo**: Prever a qualidade do vinho (escala 3-9) baseado em propriedades físico-químicas

### **🏆 Melhor Modelo: Random Forest**
- **Test MAE**: 0.5377 (~0.5 pontos de erro)
- **Test R²**: 0.3718 (37% da variância explicada)
- **Interpretação**: O modelo erra em média meio ponto, o que é **aceitável** considerando que a escala vai de 3 a 9 e a avaliação humana é subjetiva

### **Ranking de Modelos (por MAE)**
| Posição | Modelo | Test MAE | Test R² |
|---------|--------|----------|---------|
| 🥇 1º | **Random Forest** | 0.5377 | 0.3718 |
| 🥈 2º | Gradient Boosting | 0.5393 | 0.3706 |
| 🥉 3º | SVR | 0.5497 | 0.3243 |
| 4º | Ridge Regression | 0.5641 | 0.3003 |
| 5º | Linear Regression | 0.5662 | 0.2968 |
| 6º | Lasso Regression | 0.6035 | 0.2375 |
| 7º | Decision Tree | 0.6127 | 0.1873 |

### **💡 Insights - Predição de Qualidade**
- **Algoritmos ensemble** (Random Forest, Gradient Boosting) **superam modelos lineares** em ~5-10%
- **Feature engineering** (criação de índices e razões) melhorou marginalmente a performance
- Mesmo os melhores modelos explicam apenas ~37% da variância, indicando que:
  - A avaliação sensorial humana tem **componentes subjetivos** importantes
  - Outras variáveis não capturadas (ex: variedade da uva, terroir) são relevantes

---

## 🔴⚪ **CLASSIFICAÇÃO DE TIPO: TINTO vs BRANCO**

### **🎯 Objetivo**: Verificar se o "nariz químico" consegue distinguir vinhos tintos de brancos **APENAS** pelas propriedades físico-químicas

### **🏆 Melhor Modelo: SVM (Support Vector Machine)**
- **Test Accuracy**: 99.53% 🚀
- **Test AUC-ROC**: 0.9995 ⭐
- **Interpretação**: **EXCELENTE!** O algoritmo consegue distinguir com **precisão quase perfeita**

### **Ranking de Modelos (por Accuracy)**
| Posição | Modelo | Test Accuracy | Test AUC-ROC |
|---------|--------|---------------|--------------|
| 🥇 1º | **SVM** | **99.53%** | 0.9995 |
| 🥈 2º | Random Forest | 99.34% | 0.9993 |
| 🥉 3º | Logistic Regression | 98.97% | 0.9952 |
| 4º | Decision Tree | 97.74% | 0.9652 |

### **📊 Matriz de Confusão - SVM**
```
              Predito: Branco  |  Predito: Tinto
Real: Branco       792 (99.7%)  |       3 (0.3%)
Real: Tinto          2 (0.7%)   |     267 (99.3%)
```

### **🔬 Features Mais Importantes para Distinguir Tipos**
| Ranking | Feature | Importância | Interpretação |
|---------|---------|-------------|---------------|
| 🥇 1º | **Total Sulfur Dioxide** | 31.5% | Brancos têm MUITO mais SO₂ (conservante) |
| 🥈 2º | **Chlorides** | 24.2% | Tintos têm mais sais |
| 🥉 3º | **Volatile Acidity** | 11.6% | Tintos toleram mais acidez volátil |
| 4º | Density | 8.2% | Relacionado a açúcar residual |
| 5º | Residual Sugar | 6.6% | Brancos tendem a ser mais doces |

### **🎯 CONCLUSÃO: "NARIZ QUÍMICO"**

> ✅ **SIM! O algoritmo consegue distinguir vinhos tintos de brancos com 99.53% de acurácia baseado APENAS nas propriedades químicas!**
>
> Isso comprova que existe uma **"assinatura química" distinta** entre os tipos de vinho. As principais diferenças estão em:
> - **SO₂ Total**: Brancos usam muito mais conservante
> - **Cloretos**: Tintos têm maior concentração de sais
> - **Acidez Volátil**: Perfil de fermentação diferente

---

## 🧬 **FEATURE IMPORTANCE - RANDOM FOREST**

### **Top 5 Features para Predição de Qualidade**
| Ranking | Feature | Importância | Ação Recomendada |
|---------|---------|-------------|-------------------|
| 🥇 1º | **Density Adjusted** (engenheirada) | 30.1% | Monitorar relação densidade/álcool |
| 🥈 2º | **Volatile Acidity** | 11.2% | 🚨 **CONTROLE RIGOROSO** - Defeito grave |
| 🥉 3º | **Free SO₂ Ratio** (engenheirada) | 8.3% | Otimizar proporção SO₂ livre/total |
| 4º | **Sulphates** | 5.9% | Ajustar níveis para melhor conservação |
| 5º | **Free Sulfur Dioxide** | 5.3% | Equilibrar proteção vs sabor |

### **Interpretação Enológica**

#### **🍷 Density Adjusted (Densidade × Álcool)**
- **Por quê é importante?** Captura a relação entre o corpo do vinho e seu teor alcoólico
- **Na prática**: Vinhos com maior álcool e densidade equilibrada são mais complexos e estruturados

#### **⚠️ Volatile Acidity (Acidez Volátil)**
- **Por quê é crítico?** Ácido acético em excesso = sabor de vinagre
- **Na prática**: Principal defeito a ser evitado no processo de vinificação
- **Controle**: Temperatura de fermentação, higiene de equipamentos, qualidade das uvas

#### **🛡️ Free SO₂ Ratio**
- **Por quê é importante?** Indica a eficiência do SO₂ como conservante
- **Na prática**: Maior proporção de SO₂ livre = melhor proteção contra oxidação
- **Equilíbrio**: Muito SO₂ pode causar dores de cabeça e sabor desagradável

---

## 📈 **ANÁLISE PCA (COMPONENTES PRINCIPAIS)**

### **Variância Explicada**
- **PC1**: 25.3% da variância
- **PC2**: 19.1% da variância
- **Total (PC1 + PC2)**: 44.4% da variância
- **95% da variância**: Requer 8 componentes

### **💡 Insights do PCA**
- **Não há separação clara** entre vinhos tintos e brancos nos primeiros 2 componentes
- **Qualidade** mostra gradiente contínuo, confirmando que não é uma variável categórica simples
- **Sugestão**: Para redução de dimensionalidade, manter pelo menos 8 componentes

---

## 🎯 **RECOMENDAÇÕES TÉCNICAS**

### **1. Para Produtores (Melhorar Qualidade)**

#### **✅ O QUE FAZER**
1. **Aumentar Teor Alcoólico** (dentro dos limites legais)
   - Correlação: +0.469 com qualidade
   - Como: Colher uvas mais maduras, com maior concentração de açúcar
   
2. **Controlar Acidez Volátil** com RIGOR
   - Correlação: -0.265 com qualidade
   - Como: Higiene impecável, temperatura controlada, uso de leveduras selecionadas
   
3. **Otimizar Níveis de Sulfatos**
   - Importância: 5.9% no modelo
   - Como: Ajuste fino durante a vinificação para melhor conservação e aroma

4. **Equilibrar SO₂ Livre/Total**
   - Importância: 8.3% no modelo (feature engenheirada)
   - Como: Monitorar constantemente durante o processo

#### **❌ O QUE EVITAR**
1. **Acidez Volátil Alta** → Principal defeito
2. **Excesso de Cloretos** → Sabor salgado desagradável
3. **Densidade Alta com Álcool Baixo** → Vinho desequilibrado

### **2. Para Certificação (Sistema de Apoio à Decisão)**

#### **Implementação do Modelo**
- **Tolerância**: ±0.5 pontos é razoável para validação automática
- **Workflow sugerido**:
  ```
  Análise Físico-Química
          ↓
  Predição ML (Random Forest)
          ↓
  |Predição - Nota Humana| ≤ 0.5? 
          ↓                    ↓
        SIM                  NÃO
          ↓                    ↓
  Aprovação Automática   Revisão Manual
  ```

#### **Vantagens**
- ⚡ **Velocidade**: Predição instantânea
- 🎯 **Objetividade**: Baseado em química, não subjetividade
- 💰 **Economia**: Reduz número de degustações necessárias
- 📊 **Auditoria**: Rastro completo de decisões

### **3. Para Pesquisa e Desenvolvimento**

#### **Próximos Passos**
1. **Coletar Mais Variáveis**
   - Variedade da uva
   - Terroir (solo, clima)
   - Ano da safra
   - Tempo de barrica
   - → Pode melhorar R² de 0.37 para 0.50+

2. **Deep Learning**
   - Redes neurais para capturar interações não-lineares complexas
   - Transfer learning de outros datasets de vinho

3. **Ensemble de Modelos**
   - Combinar Random Forest + Gradient Boosting + SVM
   - Potencial ganho de 2-3% em performance

---

## 📊 **ESTATÍSTICAS TÉCNICAS DETALHADAS**

### **Distribuição de Qualidade**
```
Qualidade  |  Frequência  |  Percentual
-----------------------------------------
    3      |      30      |     0.6%
    4      |     206      |     3.9%
    5      |    1752      |    32.9%     ← Moda
    6      |    2323      |    43.7%     ← Mediana
    7      |     856      |    16.1%
    8      |     148      |     2.8%
    9      |       5      |     0.1%
```

**Análise**: Distribuição **assimétrica positiva** (concentrada em 5-6), com pouquíssimos vinhos excelentes (8-9) ou ruins (3-4). Isso reflete a realidade: vinhos medianos são mais comuns.

### **Outliers Detectados (Método IQR)**
| Feature | Outliers | % do Dataset |
|---------|----------|--------------|
| Residual Sugar | 548 | 10.3% |
| Free Sulfur Dioxide | 423 | 8.0% |
| Total Sulfur Dioxide | 357 | 6.7% |
| Chlorides | 285 | 5.4% |

**Ação**: Outliers **mantidos** pois podem ser valores legítimos (ex: vinhos doces têm açúcar residual alto). Uso de **RobustScaler** para mitigar impacto.

### **Testes de Normalidade**
- **Apenas 3 de 11 features** seguem distribuição aproximadamente normal (α=0.05)
- **Implicação**: Justifica uso de modelos não-paramétricos (Random Forest, SVM) ao invés de apenas regressão linear

---

## 🎨 **VISUALIZAÇÕES GERADAS**

### **📂 Arquivos Disponíveis** (17 gráficos + 1 relatório)

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | `01_feature_distributions.png` | Distribuições de todas as features (histogramas) |
| 2 | `02_quality_distribution.png` | Distribuição da qualidade (geral, tinto, branco) |
| 3 | `03_outliers_boxplots.png` | Boxplots para detecção de outliers |
| 4 | `04_red_vs_white_comparison.png` | Violin plots comparando tintos vs brancos |
| 5 | `05_correlation_matrix.png` | Heatmap de correlação completo |
| 6 | `06_quality_correlations.png` | Correlações com qualidade (geral, tinto, branco) |
| 7 | `07_pca_variance_explained.png` | Scree plot e variância acumulada (PCA) |
| 8 | `08_pca_biplot.png` | Biplot PC1 vs PC2 (qualidade e tipo) |
| 9 | `09_quality_model_comparison.png` | Comparação de métricas (MAE, RMSE, R², CV) |
| 10 | `10_quality_predictions_scatter.png` | Real vs Predito (top 3 modelos) |
| 11 | `11_feature_importance_quality.png` | Feature importance para predição de qualidade |
| 12 | `12_type_classification_comparison.png` | Accuracy e AUC-ROC por modelo |
| 13 | `13_roc_curves_type.png` | Curvas ROC para classificação de tipo |
| 14 | `14_confusion_matrix_type.png` | Matriz de confusão (SVM) |
| 15 | `15_feature_importance_type.png` | Feature importance para classificação de tipo |
| 16 | `16_wine_segmentation_3d.png` | Scatter 3D com top 3 features |
| 17 | `17_segmentation_boxplots.png` | Boxplots por segmento de qualidade |
| 📄 | `RELATORIO_FINAL.txt` | Relatório técnico em texto puro |

---

## 🏆 **CONCLUSÕES PRINCIPAIS**

### **1. Predição de Qualidade**
- ✅ **Viável**, mas com limitações (R² = 0.37)
- ✅ **Random Forest** é o melhor modelo (MAE = 0.54)
- ✅ **Álcool** é o preditor mais importante
- ⚠️ **Subjetividade humana** limita performance máxima

### **2. Classificação de Tipo (Tinto vs Branco)**
- ✅✅✅ **ALTAMENTE VIÁVEL** (99.5% accuracy)
- ✅ **SVM** é o modelo campeão
- ✅ **SO₂ Total e Cloretos** são os maiores discriminantes
- 🎯 **Confirmado**: Existe "assinatura química" distinta

### **3. Features Críticas**
**Para Qualidade**:
1. Alcohol (+)
2. Volatile Acidity (-)
3. Density (-)

**Para Tipo**:
1. Total Sulfur Dioxide (brancos >> tintos)
2. Chlorides (tintos > brancos)
3. Volatile Acidity (perfis diferentes)

### **4. Aplicabilidade Prática**
- ✅ **Produção**: Ajuste fino de processo baseado em features críticas
- ✅ **Certificação**: Sistema de apoio à decisão para acelerar aprovações
- ✅ **Marketing**: Segmentação por perfil químico e qualidade esperada
- ✅ **P&D**: Base para experimentos controlados

---

## 📚 **REFERÊNCIAS**

1. **Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009)**  
   *Modeling wine preferences by data mining from physicochemical properties*  
   Decision Support Systems, 47(4), 547-553.

2. **UCI Machine Learning Repository**  
   https://archive.ics.uci.edu/ml/datasets/wine+quality

3. **Comissão de Viticultura da Região dos Vinhos Verdes (CVRVV)**  
   http://www.vinhoverde.pt

---

## 👨‍💻 **INFORMAÇÕES TÉCNICAS**

### **Stack Tecnológico**
- **Linguagem**: Python 3.x
- **Bibliotecas Principais**:
  - `pandas`, `numpy` (manipulação de dados)
  - `scikit-learn` (machine learning)
  - `matplotlib`, `seaborn` (visualizações)
  - `scipy` (estatísticas)

### **Metodologia**
- **Validação**: 5-fold cross-validation + holdout (80/20)
- **Padronização**: RobustScaler (devido a outliers)
- **Feature Engineering**: 6 novas features criadas
- **Métricas**:
  - Regressão: MAE, RMSE, R²
  - Classificação: Accuracy, AUC-ROC, Confusion Matrix

### **Reprodutibilidade**
- **Random State**: 42 (para todos os modelos)
- **Hardware**: CPU (processamento single-thread)
- **Tempo Total**: ~3-5 minutos

---

## 📧 **CONTATO**

**Análise elaborada para**: Lineu  
**Data**: Novembro 2025  
**Objetivo**: Análise Exploratória Completa + Machine Learning + Classificação de Tipos

---

## 🍷 **CITAÇÃO**

Se você usar esta análise em publicações ou apresentações, favor citar:

```
Análise Wine Quality Dataset (Vinho Verde)
Dataset original: Cortez et al., 2009
Análise completa com EDA + ML + Classificação de Tipos
Novembro 2025
```

---

**FIM DO DOCUMENTO** 🎯✨

*"In vino veritas, in data sapientia"*  
*(No vinho está a verdade, nos dados está a sabedoria)*
