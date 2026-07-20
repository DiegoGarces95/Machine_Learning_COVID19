# COVID-19 Hospitalization Prediction

## 1. Problem Formulation and Dataset

### 1.1 Dataset
- Mexico COVID-19 clinical dataset
- 263,007 patient records, 41 columns
- Briefly explain SISVER / surveillance context
- Mention important data collection caveat:
  - severe patients were more likely to be tested
  - mild patients were sampled
  - asymptomatic people mostly absent

### 1.2 Prediction Task
- Binary classification
- Target: `TIPO_PACIENTE`
  - `1 = outpatient`
  - `2 = hospitalized`
- Recode to:
  - `0 = not hospitalized`
  - `1 = hospitalized`
- Prediction moment: initial clinical evaluation
- Prediction population: symptomatic patients represented in the dataset
- Important caveat: not a general population-level COVID risk model

### 1.3 Features and Leakage Audit
- Final features:
  - age
  - sex
  - comorbidities
  - `OTRO_CASO`
- Excluded leakage variables:
  - `NEUMONIA`
  - `INTUBADO`
  - `UCI`
  - `FECHA_DEF`
- Include a small leakage audit table

## 2. Dataset Analysis and Preprocessing

### 2.1 Target Distribution
- Include target distribution plot
- State hospitalized rate: 23.6%
- Explain why accuracy alone is misleading

### 2.2 Exploratory Data Analysis
Include 2-3 plots:
- hospitalization rate by age group
- hospitalization rate by comorbidity
- missing values in selected features

Main findings:
- hospitalization increases strongly with age
- several comorbidities are associated with higher hospitalization
- `OTRO_CASO` has many missing values

### 2.3 Preprocessing
- Fixed recoding:
  - `97/98/99 -> NaN`
  - binary variables: `1/2 -> 1/0`
- Pipeline:
  - median imputation for age
  - scaling for age
  - most frequent imputation for binary features
  - classifier
- Explain why preprocessing is inside the pipeline

### 2.4 Feature Ablation: `OTRO_CASO`
- Compare with and without `OTRO_CASO`
- Results:
  - with `OTRO_CASO`: PR-AUC 0.542, recall 0.678
  - without `OTRO_CASO`: PR-AUC 0.539, recall 0.678
- Decision: keep `OTRO_CASO`

## 3. Model Comparison

### 3.1 Experimental Setup
- 80/20 stratified train/test split
- Cross-validation only on training set
- Test set kept untouched until final evaluation
- Primary metric: PR-AUC
- Also report recall, F1, ROC-AUC

### 3.2 Models Compared
- majority baseline
- logistic regression
- random forest
- MLP

### 3.3 Cross-Validation Results
Include model comparison table:
- MLP: PR-AUC 0.550, recall 0.355
- random forest: PR-AUC 0.542, recall 0.678
- logistic regression: PR-AUC 0.511, recall 0.649
- baseline: PR-AUC 0.236, recall 0.000

Interpretation:
- MLP has slightly higher PR-AUC
- random forest has much higher recall
- choose random forest because missing hospitalized patients is costly

## 4. Final Pipeline and Hyperparameter Search

### 4.1 Final Model Choice
- Random forest
- Reason:
  - good recall
  - nonlinear relationships
  - easier to interpret than MLP

### 4.2 Hyperparameter Search
- RandomizedSearchCV
- 12 configurations
- 5-fold CV
- 60 fitted models total
- Best:
  - `n_estimators = 200`
  - `max_depth = 16`
  - `min_samples_leaf = 20`
- Best CV PR-AUC: 0.548
- Mention top configurations were very similar

## 5. Final Test Evaluation

### 5.1 Test Metrics
Report final held-out test result:
- PR-AUC: 0.549
- ROC-AUC: 0.783
- Recall: 0.703
- Precision: 0.456
- F1: 0.553
- Accuracy: 0.731

### 5.2 Confusion Matrix
Include confusion matrix:
- TN: 29,734
- FP: 10,434
- FN: 3,698
- TP: 8,736

Interpretation:
- model detects about 70% of hospitalized patients
- false positives are common
- false negatives are clinically more serious

### 5.3 Feature Importance
- Include feature importance plot/table
- Main features:
  - age
  - diabetes
  - hypertension
  - sex
  - chronic kidney disease
- Add caveat:
  - feature importance is not causal

## 6. Error Analysis, Fairness, and Limitations

### 6.1 Error Analysis
Focus on false negatives:
- overall FN rate among hospitalized patients: 29.7%
- high FN rate for:
  - age 19-30: 81.9%
  - age 31-45: 74.3%
  - no comorbidities: 55.2%
  - female hospitalized patients: 38.0% vs male 24.1%

Interpretation:
- model relies heavily on age and known comorbidities
- young or apparently healthier hospitalized patients are harder to detect

### 6.2 Fairness and Potential Harms
- Dataset is not population-level
- Severe cases are overrepresented
- Mild/asymptomatic cases underrepresented
- Some subgroups may be missed more often
- False negatives could send severe patients home
- Model should support, not replace, clinical judgment

### 6.3 Limitations
- Retrospective dataset
- Data collection bias
- Missing values, especially `OTRO_CASO`
- No oxygen saturation or symptom severity features
- No external validation

## 7. Conclusion

- Hospitalization prediction from early clinical variables is feasible
- Leakage prevention was central
- Random forest gave a useful recall-oriented model
- The model has clear subgroup weaknesses
- It should only be used as decision support

## References
- Dataset / Kaggle
- SISVER / Mexico Ministry of Health
- scikit-learn
- PR-AUC paper if used
- AI usage disclosure