# Rough project plan

(updated continuosly)

- Dataset: Mexico COVID-19 clinical dataset

- Task:
  Predict whether a patient will be hospitalized or not.
  Target variable: TIPO_PACIENTE
  1 = outpatient, 2 = hospitalized

- Features:
  Use information that should be available early, for example:
  age, sex, state/location, sector, and comorbidities like diabetes, asthma, obesity, hypertension, smoking, etc.

- Avoid leakage:
  Do not use variables that happen after hospitalization, such as ICU, intubation, death date, or the target itself.

- Split:
  Use train/test split, maybe 80/20.
  Use stratified split because the classes are imbalanced.

- Models:
  Compare a simple baseline, logistic regression, random forest, and maybe gradient boosting?

- Metrics:
  Use accuracy plus better metrics for imbalance, like F1, recall, ROC-AUC, and confusion matrix.

- EDA:
  Plot hospitalization rate by age, sex, and some comorbidities.

- Final:
  Choose the best model with cross-validation, tune it a bit, test once on the final test set, and discuss where it makes mistakes.