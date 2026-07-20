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



## 2. Dataset Analysis and Preprocessing *(Section A)*

### 2.1 Target Distribution

The target is imbalanced. In the dataset, 76.4% of patients were not hospitalized and 23.6% were hospitalized. This means that accuracy alone is not a good metric. A model that always predicts "not hospitalized" would already get high accuracy, but it would fail to detect actual hospitalized patients.

For this reason, we also use recall, F1 score, ROC-AUC, and especially PR-AUC when evaluating models.



### 2.2 Exploratory Data Analysis

The EDA shows that age is strongly related to hospitalization. Older patients have much higher hospitalization rates than younger patients.

Comorbidities are also important. Patients with known conditions have higher hospitalization rates.

Most selected features have little missing data, but `OTRO_CASO` has many unknown values. Because of this, we later do ablation study with and without it.
![Hospitalization rate by age group.](image-1.png)
![Hospitalization rate by comorbidity.](image-2.png)
![Missing values in selected features.](image-3.png)

### 2.3 Preprocessing

We applied recoding rules from the dataset documentation. Unknown codes such as `97`, `98`, and `99` were treated as missing values. For binary features, we recoded `1` as yes and `2` as no, using `1/0` coding for modeling. For `SEXO`, this means `1 = female` and `0 = male`.

The remaining preprocessing steps were placed inside a scikit-learn pipeline. Age was imputed with the median and then standardized. Binary features were imputed with the most frequent value. Keeping these steps inside the pipeline prevents information from test data from being used during training.

### 2.4 Feature Ablation: `OTRO_CASO`

`OTRO_CASO` had many missing values, so we checked whether it was meaningful to be included. We compared the same random forest model with and without this feature using cross-validation on the training set.

Keeping `OTRO_CASO` gave a slightly higher PR-AUC and F1 score, while recall stayed the same. The difference was small, but it suggested that the feature still adds some useful information. We therefore kept `OTRO_CASO` in the final feature set.



## 3. Model Comparison *(Section B)*

We first split the data into an 80% training set and a 20% test set, using stratification to keep the hospitalization rate similar in both sets. We compared four models using 5-fold stratified cross-validation on the training set. The test set was not used during this step. Each model was evaluated as a full pipeline, including preprocessing and the classifier.

Before running the experiments, we expected random forest to work well because it can capture nonlinear relationships between age and comorbidities. Logistic regression was included as a simple and interpretable model, and MLP was included as a different model family. We also used a majority-class classifier as a baseline.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.798 | 0.627 | 0.355 | 0.453 | 0.784 | 0.550 |
| Random forest | 0.742 | 0.469 | 0.678 | 0.554 | 0.781 | 0.542 |
| Logistic regression | 0.734 | 0.456 | 0.649 | 0.535 | 0.751 | 0.511 |
| Majority baseline | 0.764 | 0.000 | 0.000 | 0.000 | 0.500 | 0.236 |

The majority baseline has high accuracy because most patients were not hospitalized, but it never detects hospitalized patients. This confirms that accuracy alone is not useful for this task.

Among the real models, MLP had the highest PR-AUC, but its recall was much lower. Random forest had slightly lower PR-AUC but much higher recall. Since a false negative means missing a patient who actually needs hospitalization, random forest was considered the best model.



## 4. Final Pipeline and Hyperparameter Search *(Section C)*

### 4.1 Final Model Choice
Based on the cross-validation results, we used random forest as the final model family. 

### 4.2 Hyperparameter Search

We tuned the random forest using `RandomizedSearchCV` on the training set only. The search used 5-fold cross-validation and tested 12 hyperparameter configurations, so 60 models were fitted in total. This is within the suggested limit of about 200 fitted models.

The best selected parameters were:

| Hyperparameter | Selected value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 16 |
| `min_samples_leaf` | 20 |

The best cross-validation PR-AUC was 0.548. Several top configurations had almost the same score, so the model was not very sensitive to small changes in these hyperparameters.

## 5. Final Test Evaluation *(Section C)*

  
### 5.1 Test Metrics

After model selection and tuning, we evaluated the final random forest once on the held-out test set.

| Metric | Test score |
|---|---:|
| PR-AUC | 0.549 |
| ROC-AUC | 0.783 |
| Recall | 0.703 |
| Precision | 0.456 |
| F1 | 0.553 |
| Accuracy | 0.731 |

The model detects about 70% of hospitalized patients. Precision is lower, so the model also creates many false positives. For this task, false negatives are the more serious error.

### 5.2 Confusion Matrix


|  | Predicted not hospitalized | Predicted hospitalized |
|---|---:|---:|
| Actual not hospitalized | 29,734 | 10,434 |
| Actual hospitalized | 3,698 | 8,736 |

The model correctly detects 8,736 hospitalized patients, but misses 3,698 hospitalized patients. These false negatives are the most important errors in this task, because they correspond to patients who may need hospital care but are predicted as not hospitalized.

### 5.3 Feature Importance


We checked feature importance of the final random forest. Age has a much higher importance than the other variables, followed by diabetes and hypertension. This supports the error analysis: patients who are young or have no recorded comorbidities are harder for the model to identify as hospitalized.

| Feature | Importance |
|---|---:|
| Age | 0.549 |
| Diabetes | 0.153 |
| Hypertension | 0.095 |
| Sex | 0.053 |
| Chronic kidney disease | 0.036 |



## 6. Error Analysis and Fairness *(Section C)*

### 6.1 Error Analysis

We focused on false negatives. The overall false negative rate among hospitalized test patients was 29.7%.

The model missed younger hospitalized patients more often: the false negative rate was 81.9% for ages 19-30 and 74.3% for ages 31-45. Patients with no recorded comorbidities were also often missed, with a false negative rate of 55.2%. This suggests that the model relies strongly on age and known comorbidities.

Female hospitalized patients also had a higher false negative rate than male hospitalized patients in this test set, 38.0% compared with 24.1%. We interpret this as an observed subgroup error pattern.

### 6.2 Fairness and Potential Harms

This dataset is not population-level. Severe cases are overrepresented, while mild and asymptomatic cases are underrepresented. Testing access may also differ across regions.

The model also misses some subgroups more often, especially younger patients, patients with no recorded comorbidities, and female patients. The main harm is a false negative, which could send severe patients home.

Therefore, the model can only be used as a decision support, combined with clinical judgement from professionals.


## 7. Conclusion

This project shows that hospitalization can be predicted to some extent from early clinical information such as age, sex, comorbidities, and reported contact with another COVID-19 case. Preventing leakage was an important part of the work, because several variables in the dataset describe later hospital outcomes and should not be used.

The final random forest model detected about 70% of hospitalized patients on the test set. This is meaningful, but the error analysis showed weaknesses. Younger hospitalized patients and patients with no recorded comorbidities were more likely to be missed.

Overall, the model is best understood as a decision support tool.


## Authors
Diego Garcés, Yuqi Fang

## References

1. **Secretaría de Salud (Gobierno de México).** *Sistema de Vigilancia Epidemiológica de Enfermedades Respiratorias Virales (SISVER)*. Open Data Portal, 2020-2023.

2. **Mariana R. Franklin.** *Mexico COVID-19 Clinical Data*. Kaggle dataset. https://www.kaggle.com/datasets/marianarfranklin/mexico-covid19-clinical-data (accessed July 2026)

3. **Pedregosa, F., et al.** *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12:2825-2830, 2011.

4. **Saito, T., & Rehmsmeier, M.** *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE, 10(3):e0118432, 2015.

5. **Munzner, T.** *Visualization Analysis and Design*. CRC Press, 2014.


## AI Usage Disclaimer
LLMs was used for suggesting code structure, debugging, explaining concepts, beautifying slides, and improving wording. The final modeling decisions were reviewed by the project authors.