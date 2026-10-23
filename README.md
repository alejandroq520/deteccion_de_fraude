# Detección de Fraude en Tarjetas de Crédito — Notebook `01_analisis_inicial.ipynb`

Este repositorio contiene el notebook interactivo **`01_analisis_inicial.ipynb`**, el cual implementa un flujo completo de ciencia de datos y machine learning para la identificación de transacciones fraudulentas en tarjetas de crédito utilizando el conjunto de datos `creditcard.csv`.

---

## 📌 Resumen del Proyecto

El notebook aborda el problema de detección de fraudes caracterizado por un **desbalance extremo de clases** (fraude representa solo ~0.167% del dataset). A través de un análisis estructurado y reproducible, se estudian la calidad del dato, el comportamiento de las variables, la relevancia de la variable temporal, la reducción de dimensionalidad a un subconjunto clave y la optimización de hiperparámetros mediante validación cruzada estratificada.

---

## 📂 Datos Utilizados

- **Archivo**: `creditcard.csv`
- **Volumen inicial**: 284,807 transacciones y 31 columnas.
- **Variables**:
  - `Time`: Segundos transcurridos entre cada transacción y la primera transacción del dataset.
  - `V1` a `V28`: Características numéricas anonimizadas resultantes de una transformación previa por Análisis de Componentes Principales (PCA).
  - `Amount`: Importe monetario de la transacción.
  - `Class`: Variable objetivo binaria (`0` = Transacción legítima, `1` = Fraude).

---

## 🔬 Estructura y Flujo del Notebook

El notebook está organizado en las siguientes secciones analíticas:

### 1. Imports y Configuración
Carga de librerías esenciales para manipulación de datos (`pandas`, `numpy`), visualización (`matplotlib`, `seaborn`) y modelado (`scikit-learn`, `xgboost`).

### 2. Carga y Vista Inicial
Inspección inicial de dimensiones, primeras filas e integridad de tipos de datos.

### 3. Calidad de Datos y Limpieza
- **Valores Nulos**: Se verifica la ausencia total de valores nulos o faltantes (0 nulos).
- **Duplicados**: Se detectan **1,081 registros duplicados**, los cuales se eliminan para evitar sesgos de sobreajuste, dejando **283,726 transacciones únicas**.
- **Distribución tras limpieza**:
  - Transacciones legítimas (`Class = 0`): 283,253 (99.833%)
  - Transacciones fraudulentas (`Class = 1`): 473 (0.167%)
- **Criterio sobre Outliers**: Se analiza la regla del rango intercuartílico (IQR) y puntajes Z ($z \ge 3$). Se concluye **no eliminar outliers** ya que descartarían la mayor parte de las observaciones críticas y los propios patrones de fraude.

### 4. Análisis Exploratorio de Datos (EDA)
- **Análisis univariado y distribuciones**: Inspección visual de la distribución y dispersión de cada variable respecto a la clase mediante scatter plots con dispersión controlada (*jitter*).
- **Correlación entre variables**: Al provenir la mayoría de PCA, las características $V_1$ a $V_{28}$ son prácticamente ortogonales entre sí, existiendo únicamente correlaciones moderadas aisladas (ej. `Amount` y `V2` con -0.53).
- **Variables discriminantes identificadas visualmente**: $V_2$, $V_{10}$, $V_{11}$, $V_{12}$, $V_{14}$, $V_{17}$ y $V_{27}$.

### 5. Aporte y Evaluación de la Variable `Time`
Se evalúa formalmente el impacto predictivo de la variable temporal comparando tres variantes:
1. `Time` en su escala original.
2. `Time` con transformación cíclica sinusoidal de 24 horas (`Time_sin`, `Time_cos` con período de 86,400 segundos).
3. Eliminación completa de `Time`.

**Hallazgo**: La variable `Time` (tanto original como cíclica) no aporta mejoras apreciables ni consistentes en el PR-AUC o F1 de los modelos (XGBoost, Random Forest, Regresión Logística). Por ende, se decide prescindir de ella para simplificar el pipeline.

### 6. Selección de Características: 29 Variables vs 8 Seleccionadas
Se compara el desempeño de entrenar con todas las variables numéricas (28 componentes PCA + `Amount`) frente a un subconjunto compacto de 8 variables seleccionadas:
$$\text{Variables Seleccionadas} = [V_2, V_{10}, V_{11}, V_{12}, V_{14}, V_{17}, V_{27}, \text{Amount}]$$

**Hallazgo**: El subconjunto de 8 variables retiene prácticamente todo el poder predictivo con una reducción de más del 72% en el espacio de características.

### 7. Partición Estratificada de Datos
- División 80% entrenamiento (226,980 transacciones, 378 fraudes) y 20% prueba (56,746 transacciones, 95 fraudes).
- Se utiliza partición estratificada (`stratify=y`) con semilla fija (`random_state=42`) para garantizar reproducibilidad exacta y mantener constante la proporción de fraudes.

### 8. Optimización de Hiperparámetros con Validación Cruzada
Se implementa `GridSearchCV` con `StratifiedKFold` (5 particiones), optimizando específicamente **PR-AUC** (*Average Precision*), que es la métrica adecuada ante el desbalance severo:

- **Regresión Logística**: Escalamiento con `StandardScaler`, ajuste de regularización $C$ y ponderación de clases `class_weight`.
- **Random Forest Classifier**: Búsqueda en número de estimadores, profundidad máxima (`max_depth`), muestras mínimas por hoja y balanceo.
- **XGBoost Classifier**: Búsqueda en tasa de aprendizaje (`learning_rate`), profundidad de árboles (`max_depth`), estimadores y `scale_pos_weight`.

---

## 📊 Resultados y Comparativa de Modelos

Resultados obtenidos en el conjunto de prueba (Test: 56,746 muestras, 95 fraudes) sobre las **8 variables seleccionadas**:

| Modelo | PR-AUC (CV) | PR-AUC (Prueba) | ROC-AUC (Prueba) | Precision (Prueba) | Recall (Prueba) | F1-Score (Prueba) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.8347** | **0.7750** | **0.9629** | **0.9211** | **0.7368** | **0.8187** |
| **XGBoost** | 0.8184 | 0.7634 | 0.9675 | 0.9079 | 0.7263 | 0.8070 |
| **Regresión Logística** | 0.7424 | 0.6438 | 0.9654 | 0.8714 | 0.6421 | 0.7394 |

### Mejores Parámetros Encontrados:
- **Random Forest**: `class_weight: None`, `max_depth: 12`, `min_samples_leaf: 1`, `n_estimators: 200`
- **XGBoost**: `learning_rate: 0.05`, `max_depth: 3`, `n_estimators: 200`, `scale_pos_weight: 1`
- **Regresión Logística**: `C: 0.1`, `class_weight: None`

---

## 💡 Conclusiones del Estudio

1. **Eficiencia con Menos Características**: Reducir el conjunto de entrada de 29 a 8 variables predictoras clave (`V2, V10, V11, V12, V14, V17, V27, Amount`) mantiene un rendimiento competitivo (F1 > 0.81), disminuyendo la complejidad y el coste computacional.
2. **Modelo Recomendado**: **Random Forest** demostró ser el modelo más robusto, alcanzando un **PR-AUC de 0.7750**, una **Precisión de 92.11%** (mínimos falsos positivos) y un **Recall de 73.68%**.
3. **Métrica de Negocio**: En problemas de detección de fraude con proporciones de 0.167%, métricas como ROC-AUC suelen resultar optimistas (~0.96-0.97) debido al volumen masivo de verdaderos negativos; el **PR-AUC** y el **F1-score** proporcionan una evaluación realista del rendimiento.

---

## 🛠 Requisitos y Ejecución

Para ejecutar el notebook localmente:

1. Asegúrate de tener Python 3.9+ instalado con las siguientes dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```
2. Coloca `creditcard.csv` en el mismo directorio que el notebook.
3. Inicia Jupyter Lab o Notebook:
   ```bash
   jupyter notebook 01_analisis_inicial.ipynb
   ```
