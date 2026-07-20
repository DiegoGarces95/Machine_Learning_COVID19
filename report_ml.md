# COVID-19 Hospitalization Prediction

## 1. Problem Formulation and Dataset *(Section A)*

### 1.1 Dataset

We use the Mexico COVID-19 Clinical Data dataset. The version used here contains 263,007 patient records and 41 columns. Each row represents one patient record, including dates, demographic information, comorbidities, and hospitalization-related variables.

The dataset is appropriate for this project because it contains early patient information, such as age, sex, and comorbidities, as well as the outcome we want to predict: whether the patient was treated as an outpatient or hospitalized.

A key limitation is that the dataset is not a random sample of all COVID-19 infections. Patients with severe symptoms were more likely to be tested, while only 10% mild-symptom patients were tested, and asymptomatic people were mostly never tested. Therefore, the results mainly apply to patients represented in this surveillance dataset, rather than all people infected with COVID-19.



### 1.2 Prediction Task

We formulate the project as a binary classification task. The target variable is `TIPO_PACIENTE`, where `1` means outpatient care and `2` means hospitalization. For modeling, we recode this into a binary variable called `hospitalized`, where `0` means not hospitalized and `1` means hospitalized.

The prediction moment is the initial clinical evaluation. This means the model should only use information known when the patient first arrives, such as age, sex, and known comorbidities. The prediction population is therefore patients similar to those represented in the surveillance dataset.



### 1.3 Features and Leakage Audit

The final feature set includes age, sex, several comorbidities, and whether the patient had contact with another COVID-19 case (`OTRO_CASO`).

The comorbidity variables include diabetes, COPD, asthma, immunosuppression, hypertension, cardiovascular disease, obesity, chronic kidney disease, other comorbidities, and smoking. These variables are useful because they describe the patient's risk factors available at the time of evaluation.

We excluded variables that could leak information from later stages of the hospitalization. In particular, we did not use `INTUBADO`, `UCI`, `FECHA_DEF`, or `NEUMONIA`. Intubation, ICU admission, and death date are clearly downstream outcomes. `NEUMONIA` was also excluded because it may be recorded either first-minute or during hospitalization.
| Variable | Available at prediction time? | Included? | Reason |
|---|---:|---:|---|
| `EDAD` | Yes | Yes | Baseline demographic information |
| `SEXO` | Yes | Yes | Baseline demographic information |
| Comorbidities | Yes | Yes | Usually known from patient history |
| `OTRO_CASO` | Probably | Yes | Exposure history may be asked at intake |
| `NEUMONIA` | Unclear | No | May be diagnosed later and could cause leakage |
| `INTUBADO` | No | No | Downstream hospital procedure |
| `UCI` | No | No | ICU admission happens after initial triage |
| `FECHA_DEF` | No | No | Death date is a future outcome |

### ❓❓❓1.4 Representation and Fairness Considerations
- Who is represented in this dataset in the first place is not neutral: testing was rationed by severity (100% of severe/SARI cases, ~10% of mild cases sampled, asymptomatic cases essentially absent), so access to testing itself may vary by region and socioeconomic access to care, not only by clinical severity
- Consider age and sex balance in the raw data before modeling: are any age groups or sexes over/under-represented relative to what we'd expect in the general symptomatic population?
- Flag upfront that these representation gaps could translate into systematically worse model performance for under-represented subgroups
- Note this is revisited with empirical evidence (false-negative rate by subgroup) in Section 6.2, after the final model is evaluated

## 2. Dataset Analysis and Preprocessing *(Section A, cont.)*

### 2.1 Target Distribution

The target is imbalanced. In the dataset, 76.4% of patients were not hospitalized and 23.6% were hospitalized. This means that accuracy alone is not a good metric. A model that always predicts "not hospitalized" would already get high accuracy, but it would fail to detect actual hospitalized patients.

For this reason, we also use recall, F1 score, ROC-AUC, and especially PR-AUC when evaluating models.
![alt text](image.png)


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

## 3. Model Comparison *(Section B)*

### 3.1 Experimental Setup
- 80/20 stratified train/test split
- Cross-validation only on training set
- Test set kept untouched until final evaluation
- Primary metric: PR-AUC
- Also report recall, F1, ROC-AUC

### 3.2 Intuition (stated before running experiments)
- Expect random forest to perform well: it can capture nonlinear interactions between age and comorbidities that a linear model would miss
- Expect logistic regression to be easier to interpret but potentially miss those interactions
- MLP included as a different model family, with no strong prior on whether it will outperform random forest on this tabular dataset

### 3.3 Models Compared
- majority baseline
- logistic regression
- random forest
- MLP

### 3.4 Cross-Validation Results
Include model comparison table:
- MLP: PR-AUC 0.550, recall 0.355
- random forest: PR-AUC 0.542, recall 0.678
- logistic regression: PR-AUC 0.511, recall 0.649
- baseline: PR-AUC 0.236, recall 0.000

Interpretation:
- MLP has slightly higher PR-AUC than expected relative to random forest — a mild surprise worth one sentence of explanation (e.g., MLP may fit the majority class distribution slightly better on average without being better at ranking the minority class high, since recall tells a different story)
- random forest has much higher recall, consistent with the intuition above
- choose random forest because missing hospitalized patients is costly, so recall is prioritized over the single highest PR-AUC

## 4. Final Pipeline and Hyperparameter Search *(Section C)*

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

## 5. Final Test Evaluation *(Section C, cont.)*

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

## 6. Error Analysis, Fairness, and Limitations *(Section C, cont.)*

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
- As anticipated in Section 1.4, the dataset is not population-level: severe cases are overrepresented, mild/asymptomatic cases underrepresented, and testing access itself may vary by region and socioeconomic status
- The error analysis above confirms the representation concern translates into uneven model performance: some subgroups (younger adults, patients with no recorded comorbidities, female hospitalized patients) are missed more often than others
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