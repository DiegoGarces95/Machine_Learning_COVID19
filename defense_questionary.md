# 🎓 Master's Defense Questionary
## Machine Learning for COVID-19 Clinical Triage
### Project: Minute 1 Hospitalization Prediction — SISVER Dataset

> **Scope:** Questions a professor or thesis committee is likely to ask during your defense, organized by topic.
> Each question includes the **expected answer** directly tied to your code and paper.

---

## 📦 SECTION 1 — Data & Problem Formulation

---

### Q1. Why did you use only 263,007 patients from 3.8 million records?

**Expected Answer:**
The full SISVER registry contains ~3.8M historical records spanning all respiratory viruses and test results.
We first filter to **confirmed COVID-19 positive cases only**.
Then we draw a **stratified random sample** (`stratify=y`) to obtain a statistically representative subset that preserves the 3.2:1 class imbalance, which allows fast, reproducible experimentation without compromising statistical fidelity.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y   # Preserves the 76.4% / 23.6% ratio
)
```

---

### Q2. How exactly is the binary target defined? What does "1" mean?

**Expected Answer:**
The target `hospitalized` is derived from the original `TIPO_PACIENTE` column:

```python
df["hospitalized"] = (df["TIPO_PACIENTE"] == 2).astype(int)
```

- **0 = Outpatient / Mild** — patient managed ambulatorily at home (76.4%).
- **1 = Hospitalized / Severe** — patient required immediate inpatient admission (23.6%).

This is a **binary classification** problem. The minority positive class (1) is the medically critical one.

---

### Q3. What is data leakage and why is it so critical here?

**Expected Answer:**
Data leakage occurs when information that would **not be available at prediction time** is used as a model input, causing artificially inflated metrics that collapse in production.

In this clinical context, the prediction must be made at **Minute 1** of patient intake. Variables like `INTUBADO`, `UCI`, `FECHA_DEF` (death date), `RESULTADO` (PCR result, takes 24–72 hours), and `NEUMONIA` (diagnosed via imaging inside the hospital) are **all post-triage outcomes** and were explicitly excluded.

```python
leakage_audit = pd.DataFrame([
    {"variable": "UCI",       "available_at_prediction_time": "no",      "included": "no"},
    {"variable": "INTUBADO",  "available_at_prediction_time": "no",      "included": "no"},
    {"variable": "FECHA_DEF", "available_at_prediction_time": "no",      "included": "no"},
    {"variable": "RESULTADO", "available_at_prediction_time": "no",      "included": "no"},
    {"variable": "NEUMONIA",  "available_at_prediction_time": "unclear", "included": "no for base model"},
])
```

Including any of these could push accuracy toward ~100% on training data, but the model would be **useless** in real deployment.

---

### Q4. Why did you include NEUMONIA in the suspicious features but exclude it?

**Expected Answer:**
`NEUMONIA` is ambiguous: in principle, pneumonia can be clinically suspected on arrival.
However, in the SISVER coding conventions, pneumonia entries reflect **hospital-diagnosed** cases (chest X-ray or CT imaging performed *inside* the facility after admission). This makes it **indistinguishable from a post-triage complication**.

We conducted an **ablation study** on `OTRO_CASO` (the borderline feature), and applied similar reasoning to exclude NEUMONIA from the base model to maintain strict clinical realism.

---

### Q5. You have 13 final features. Which ones and why?

**Expected Answer:**
The final feature set is:

| Feature | Type | Justification |
|---|---|---|
| `EDAD` | Numeric | Age is the dominant hospitalization risk factor |
| `SEXO` | Binary | Males show +3.4% higher hospitalization rate |
| `DIABETES` | Comorbidity | 2nd most important feature (14.8% importance) |
| `HIPERTENSION` | Comorbidity | 3rd most important (9.3% importance) |
| `OBESIDAD` | Comorbidity | Known COVID-19 severity multiplier |
| `EPOC` | Comorbidity | COPD patients: 62.7% hospitalization rate |
| `ASMA` | Comorbidity | Pre-existing respiratory condition |
| `RENAL_CRONICA` | Comorbidity | 66.8% hospitalization rate — highest of all |
| `CARDIOVASCULAR` | Comorbidity | Known cardiac risk factor |
| `INMUSUPR` | Comorbidity | 54.9% hospitalization rate |
| `TABAQUISMO` | Risk factor | Smoking damages respiratory system |
| `OTRA_COM` | Comorbidity | Catch-all for other chronic conditions |
| `OTRO_CASO` | Exposure | COVID contact — validated via ablation study |

All 13 are **available at Minute 1** of triage intake.

---

## 📊 SECTION 2 — Class Imbalance & Metric Selection

---

### Q6. Why is accuracy a misleading metric for this dataset?

**Expected Answer:**
With a **3.2:1 class imbalance** (76.4% outpatients vs. 23.6% hospitalized), a naive model that predicts the majority class for *every* patient achieves:

- **Accuracy = 76.36%** ← sounds good but is clinically catastrophic
- **Recall = 0.000** ← misses *every single* hospitalized patient
- **PR-AUC = 0.2364** ← essentially random for the minority class

```python
baseline_clf = DummyClassifier(strategy="most_frequent")
```

This is demonstrated by the `DummyClassifier` baseline. High accuracy while detecting zero real emergencies shows accuracy is useless for imbalanced triage scenarios.

---

### Q7. What is PR-AUC and why did you choose it over ROC-AUC?

**Expected Answer:**

- **ROC-AUC** measures discrimination at all thresholds using TPR vs. FPR. It can be **optimistic on imbalanced datasets** because a high TN count inflates the FPR denominator, making the curve look artificially good.
- **PR-AUC (Average Precision)** measures the area under the Precision-Recall curve, which focuses **exclusively on the positive (minority) class**. It is more informative when the positive class is rare and medically critical.

The dummy classifier has:
- ROC-AUC = 0.5 (looks "random")
- PR-AUC = 0.2364 (equals the prevalence rate — the true floor)

Our best model:
- ROC-AUC = 0.8044
- PR-AUC = 0.5771 (+132% over dummy)

Reference: *Saito & Rehmsmeier, PLOS ONE, 2015* — cited in the paper.

---

### Q8. How did you handle class imbalance in the models?

**Expected Answer:**
Two complementary techniques:

1. **`class_weight='balanced'`** in scikit-learn classifiers — automatically adjusts internal loss/split weights inversely proportional to class frequency:

```python
rf_clf = RandomForestClassifier(
    class_weight="balanced",  # weights: n_samples / (n_classes * bincount)
    ...
)
logreg_clf = LogisticRegression(
    class_weight="balanced",
    ...
)
```

2. **Optimizing PR-AUC directly** during hyperparameter search (`scoring="average_precision"`), not accuracy.

We deliberately did **not** use SMOTE to keep the preprocessing pipeline clean and avoid synthetic patient generation on medical data.

---

## 🧪 SECTION 3 — Preprocessing & Pipeline Architecture

---

### Q9. Why do you put all preprocessing inside a scikit-learn Pipeline?

**Expected Answer:**
To guarantee **zero data leakage across cross-validation folds**. If we impute or scale *before* splitting, the imputer learns statistics (median, mode) from the entire dataset including the validation folds, which is leakage.

Inside a `Pipeline`, the `fit()` of the preprocessor only sees the **training fold** in each cross-validation iteration:

```python
model_pipelines[model_name] = Pipeline([
    ("preprocessor", preprocessor),   # fitted only on train fold
    ("classifier", clf)
])
```

The `ColumnTransformer` combines numeric and binary processing in one atomic step.

---

### Q10. Why median imputation for age and not mean? And why constant=0 for comorbidities?

**Expected Answer:**

**For `EDAD` (age):**
- Age distributions are often **right-skewed** (few very old patients pull the mean up).
- Median is more robust to outliers and imputes to the "typical" patient.

```python
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

**For comorbidities — ablation study result:**

| Imputation Strategy | CV PR-AUC |
|---|---|
| `most_frequent` (mode) | 0.5109 ± 0.0027 |
| `constant = 0` | **0.5601 ± 0.0011** |

Clinical rationale: in emergency intake documentation, an **unrecorded comorbidity overwhelmingly means it is absent**, not positive. Zero-fill better represents clinical reality.

```python
binary_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0))
])
```

---

### Q11. Why do you apply StandardScaler only to age and not to the binary features?

**Expected Answer:**
`StandardScaler` converts values to Z-scores (mean=0, std=1). This is useful for:
- **Logistic Regression** — feature scaling ensures gradient descent converges efficiently.
- **MLP** — neural networks are sensitive to feature magnitude.

Binary features (0/1) are already on the **same scale** by definition — scaling them would destroy interpretability without improving performance. Tree-based models (Random Forest) are **scale-invariant** anyway.

---

### Q12. What does `SEXO` = 2 mean in the original data, and what did you do with it?

**Expected Answer:**
In the SISVER coding:
- `SEXO = 1` → Female
- `SEXO = 2` → Male
- `SEXO = 97/98/99` → Missing/Unknown

Recoding applied to all binary features:

```python
unknown_codes = [97, 98, 99]
for col in binary_features:
    model_df[col] = model_df[col].replace(unknown_codes, np.nan)
    model_df[col] = model_df[col].replace({2: 0})
```

This creates a consistent `0/1/NaN` encoding across all binary features before pipeline imputation.

---

## 🌲 SECTION 4 — Model Selection & Hyperparameter Tuning

---

### Q13. You benchmarked 5 models. Why did you choose Random Forest as your final model?

**Expected Answer:**
Cross-validation benchmark:

| Model | PR-AUC | Recall |
|---|---|---|
| Gradient Boosting | 0.5532 | 0.357 |
| MLP | 0.5492 | 0.347 |
| Logistic Regression | 0.5109 | 0.650 |
| **Random Forest (base)** | 0.4945 | 0.622 |
| DummyClassifier | 0.2364 | 0.000 |

Random Forest was selected because:
1. Built-in **`class_weight='balanced'`** support → 62%+ recall at baseline.
2. **Native feature importances** → clinical interpretability requirement.
3. Highly **amenable to hyperparameter tuning**.
4. After tuning: **PR-AUC = 0.5771**, competitive with Gradient Boosting.
5. Trees are **scale-invariant**, reducing preprocessing sensitivity.

---

### Q14. What is `RandomizedSearchCV` and why use it instead of `GridSearchCV`?

**Expected Answer:**
`GridSearchCV` exhaustively tries **all combinations**: 3×4×3 = 36 combinations × 5 folds = **180 model fits**.

`RandomizedSearchCV` samples `n_iter=12` random combinations → **60 model fits** (~3× faster).

```python
rf_search = RandomizedSearchCV(
    estimator=final_rf_pipeline,
    param_distributions=param_dist,
    n_iter=12,
    scoring="average_precision",   # optimize PR-AUC, not accuracy
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_search.fit(X_train, y_train)   # only trained on X_train, never X_test
```

The key: `rf_search.fit()` is called **only on `X_train`**. `X_test` remains locked.

---

### Q15. What do `max_depth`, `min_samples_leaf`, and `n_estimators` control?

**Expected Answer:**

| Parameter | What it Controls | Too High → | Too Low → |
|---|---|---|---|
| `n_estimators` | Number of trees | Slower, diminishing returns | High variance |
| `max_depth` | Maximum depth per tree | Overfitting | Underfitting |
| `min_samples_leaf` | Minimum samples at a leaf | Underfitting | Overfitting |

Best configuration found:
`n_estimators=150`, `max_depth=15`, `min_samples_leaf=10`, `min_samples_split=20`.

---

### Q16. How does `class_weight='balanced'` work mathematically?

**Expected Answer:**
Scikit-learn computes per-class weights as:

```
w_k = n_samples / (n_classes × n_samples_k)
```

With n=210,405, n_classes=2:
- **Class 0 (not hospitalized):** 76.4% → n≈160,753 → weight ≈ 0.655
- **Class 1 (hospitalized):** 23.6% → n≈49,652 → weight ≈ **2.119**

Each hospitalized patient counts ~3.2× more when evaluating splits, directly compensating for the imbalance.

---

## 📈 SECTION 5 — Evaluation & Results Interpretation

---

### Q17. Walk me through the confusion matrix of your best model.

**Expected Answer:**

|  | Predicted Not Hospitalized | Predicted Hospitalized |
|---|---|---|
| **Actual Not Hospitalized** | TN = 29,898 | FP = 10,286 |
| **Actual Hospitalized** | FN = 3,728 | TP = 8,688 |

Clinical interpretation:
- **TP (8,688):** Correctly flagged → rapid admission → **saves lives**.
- **FN (3,728):** Missed severe cases sent home → **highest clinical cost** (potential death).
- **FP (10,286):** Mild patients held for observation → **costly but safe**.
- **TN (29,898):** Correctly discharged → **decongests emergency rooms**.

**Recall** = 8,688 / (8,688 + 3,728) = **70.1%**  
**Precision** = 8,688 / (8,688 + 10,286) = **45.8%**

---

### Q18. Why is Recall more important than Precision in this clinical context?

**Expected Answer:**
The **asymmetry of errors**:

- **False Negative** (missed hospitalized patient) → Discharged home → Risk of respiratory failure and death.
- **False Positive** (mild patient flagged as severe) → Extra observation → Increased cost, but **no patient harm**.

In triage for life-threatening conditions, **missing a severe case is far worse than a false alarm**. Hence, we prioritize maximizing Recall (sensitivity) over Precision (PPV).

---

### Q19. What does PR-AUC of 0.5771 actually mean? Is it "good"?

**Expected Answer:**

| Benchmark | PR-AUC |
|---|---|
| Random/Dummy (floor) | 0.2364 |
| **Our model** | **0.5771** |
| Perfect model | 1.0000 |

**+132% above baseline.** Whether it is "good" depends on context:
- Only 13 features are available (no vitals, no imaging, no symptoms).
- Prediction at **Minute 1** — before clinical assessment.
- This is the fundamental upper bound of predicting from basic demographics + comorbidities.

The model's value is providing a **probability score** that triggers differential triage protocols, not replacing clinical judgment.

---

### Q20. How do you interpret feature importances in a Random Forest?

**Expected Answer:**
Scikit-learn's `feature_importances_` reports **Mean Decrease in Impurity (MDI)**:

```python
importances = best_model.named_steps["classifier"].feature_importances_
feature_importance_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
```

Results:

| Feature | Importance |
|---|---|
| EDAD (age) | **55.5%** |
| DIABETES | 14.8% |
| HIPERTENSION | 9.3% |
| SEXO | 5.2% |
| RENAL_CRONICA | 3.5% |

**Limitation:** MDI can be **biased toward high-cardinality features** (like EDAD). Permutation importance is more robust but computationally expensive.

---

## 🔍 SECTION 6 — Subgroup Error Analysis & Fairness

---

### Q21. Your model has 70.1% overall Recall. Is it equally accurate across all patients?

**Expected Answer:**
**No — this is one of the most critical findings.**

```python
error_df["is_fn"] = ((error_df["actual"] == 1) & (error_df["predicted"] == 0)).astype(int)
hospitalized_df = error_df[error_df["actual"] == 1].copy()
fn_by_age = hospitalized_df.groupby("age_group")["is_fn"].agg(["mean", "count"])
```

| Subgroup | FN Rate |
|---|---|
| Age 20–39 | **82.0%** ← critical blind spot |
| Age 46–60 | ~25% |
| Age 61–75 | ~15% |
| Female patients | 31.6% |
| Male patients | 28.9% |
| Zero comorbidities | **53.9%** |
| 3+ comorbidities | 16.7% |

Two structural vulnerabilities: **The Young Adult Paradox** and **The Comorbidity Absence Blind Spot**.

---

### Q22. Why do young adults (20–39) have an 82% False Negative rate?

**Expected Answer:**
The model assigns **55.5% predictive weight to age**. In the training data, patients <40 years old have hospitalization rates **below 15%**. The model learns: "young patient → probably mild → predict outpatient."

When a young patient deteriorates (early acute hypoxia, viral myocarditis), they arrive **without chronic illnesses** and with a young age — both features strongly associated with "outpatient."

**Operational safeguard:** For patients with p̂ < 0.50 AND age < 40 AND zero comorbidities → **mandatory pulse oximetry**. If SpO₂ < 92%, override the ML outpatient prediction.

---

### Q23. What is the Comorbidity Absence Blind Spot?

**Expected Answer:**
The model uses comorbidities as risk multipliers. When a patient has **zero comorbidities**, these features contribute nothing, and prediction defaults toward ambulatory discharge.

- **0 comorbidities → FN rate = 53.9%**
- **3+ comorbidities → FN rate = 16.7%**

COVID-19 can be severe in previously healthy individuals (inflammatory storm, acute hypoxemia), but the model's training signal heavily penalizes the absence of known risk factors.

---

### Q24. Is this model "fair" across demographic groups?

**Expected Answer:**
The model exhibits **differential error rates** across subgroups — an algorithmic fairness concern:

1. **Age bias:** Elderly patients are over-represented in high-severity training signal → model performs worse for young patients.
2. **Geographic variance:** Hospitalization rates range from 10.9% to 41.4% across federal entities — reflecting local hospital capacity, not biological risk.
3. **Sampling bias:** SISVER prioritized testing severely ill patients during peak waves → outpatients in the registry are more clinically compromised than typical primary care patients.

All these are explicitly documented in Section 3.3 of the paper with supporting visualizations.

---

## 🏗️ SECTION 7 — Methodology & Code Design

---

### Q25. What is StratifiedKFold and why use it instead of regular KFold?

**Expected Answer:**
`StratifiedKFold` ensures each fold has **the same class distribution** as the full dataset:

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)
```

With plain `KFold`, a fold could by chance contain 30% hospitalized patients, skewing all metric calculations. With stratification, every fold maintains ~23.6% hospitalized / ~76.4% not hospitalized, making PR-AUC estimates **unbiased and stable**.

---

### Q26. Why do you call `cross_validate` on `X_train` only and never touch `X_test` until the end?

**Expected Answer:**
This is the **golden rule of ML evaluation**: the test set must simulate completely unseen data.

If `X_test` is used during tuning → **test set contamination**.  
If `X_test` is used for feature selection → **selection bias**.

```python
# WRONG: rf_search.fit(X, y)        # Uses test data
# RIGHT:
rf_search.fit(X_train, y_train)     # Only training data
y_test_pred = best_model.predict(X_test)  # Locked vault, final step only
```

All CV and hyperparameter search uses exclusively `X_train` (210,405 patients).
`X_test` (52,602 patients) is unlocked **once** at the very end.

---

### Q27. What does `best_estimator_` return from `RandomizedSearchCV`? Is it refitted?

**Expected Answer:**
By default, `RandomizedSearchCV` has `refit=True`. After finding best hyperparameters via CV, it **automatically refits on the entire training set**:

```python
best_model = rf_search.best_estimator_
# This pipeline was refit on X_train (n=210,405) with optimal hyperparameters
y_test_pred = best_model.predict(X_test)
```

The final model benefits from **all 210,405 training samples**, not just the training folds used during search.

---

### Q28. How did you extract feature importances from a Pipeline? Why `named_steps`?

**Expected Answer:**
A `Pipeline` wraps steps under named keys. `named_steps` allows access by string name:

```python
importances = best_model.named_steps["classifier"].feature_importances_
```

`best_model` is a `Pipeline` with:
1. `"preprocessor"` → `ColumnTransformer`
2. `"classifier"` → `RandomForestClassifier`

`feature_importances_` belongs to the `RandomForestClassifier` (step 2). The feature order must match `ColumnTransformer` output (numeric first, then binary):

```python
feature_names = numeric_features + binary_features  # ["EDAD", "SEXO", "DIABETES", ...]
```

---

## 💡 SECTION 8 — Critical Thinking & Extensions

---

### Q29. What would happen if you included `NEUMONIA` in your model?

**Expected Answer:**
Recall and PR-AUC would increase dramatically (possibly PR-AUC > 0.80). However, this would be **severe data leakage** — in SISVER, pneumonia is coded after hospital imaging.

The model would **collapse in real triage** because at Minute 1, you don't yet have a pneumonia diagnosis.

This is the *retrospective dataset trap*: perfect in analysis, useless in deployment.

---

### Q30. What alternative techniques could improve performance?

**Expected Answer — methodological depth:**

1. **XGBoost/LightGBM:** Already benchmarked at CV PR-AUC = 0.5532 without tuning. Full search would likely outperform RF.
2. **SMOTE/ADASYN:** Synthetic oversampling — must be applied **inside the pipeline** after train/val split to avoid leakage.
3. **Threshold optimization:** Find the optimal decision threshold on the validation PR curve to maximize Recall at acceptable Precision, instead of using the default 0.5.
4. **Additional features:** Vital signs (SpO₂, temperature, heart rate) at intake would dramatically improve prediction. The model is constrained by SISVER's administrative data structure.
5. **Calibration:** `CalibratedClassifierCV` to produce well-calibrated probability scores for clinical use.
6. **Explainability:** SHAP values (per-patient) are more appropriate for clinical deployment than MDI feature importances.

---

### Q31. Would you deploy this model in a real hospital? What would be required?

**Expected Answer — expected mature, nuanced answer:**

**Not directly** — additional requirements:

1. **Prospective validation:** Retrospective SISVER performance does not guarantee prospective triage performance (distribution shift, new variants, changing hospital protocols).
2. **Regulatory approval:** Medical AI requires clearance (FDA 510k, CE marking in EU, COFEPRIS in Mexico).
3. **Calibration:** Probability outputs need calibration to be clinically interpretable.
4. **Human-in-the-loop:** The model provides a **decision support score**, not an autonomous decision. Clinical override rules are mandatory.
5. **Monitoring:** Model performance must be monitored continuously as variants and protocols evolve (concept drift).
6. **Fairness remediation:** The 82% FN rate for young adults must be addressed before deployment.

The proposed **3-track triage protocol** is precisely designed for safe ML augmentation:

| Track | Condition | Action |
|---|---|---|
| High-Risk | p̂ ≥ 0.50 | Immediate priority admission |
| Ambulatory | p̂ < 0.50, Age ≥ 40, or comorbidities | Standard outpatient discharge |
| Clinical Safeguard | p̂ < 0.50, Age < 40, zero comorbidities | **Mandatory SpO₂** → override if < 92% |

---

### Q32. What is the significance of the +132% improvement over the dummy baseline?

**Expected Answer:**
The dummy classifier always predicts "outpatient" and achieves:
- PR-AUC = 0.2364 (equal to the prevalence — the theoretical floor)
- Recall = 0.000 (zero hospitalized patients caught)

Our model: PR-AUC = 0.5771.

The key takeaway: **any model that does not beat this baseline adds zero clinical value**, and reporting only accuracy (76.36% for the baseline) would falsely suggest a good model where there is none.

---

## 🔢 SECTION 9 — Code Snippet Questions (Live Defense)

---

### Q33. What does this code do? What could go wrong?

```python
for col in binary_features:
    model_df[col] = model_df[col].replace(unknown_codes, np.nan)
    model_df[col] = model_df[col].replace({2: 0})
```

**Expected Answer:**
1. Replaces SISVER unknown codes (97, 98, 99) with `NaN` — marking them for imputation.
2. Recodes `2 → 0` — converting the "no" / "male" response to 0 for all binary variables.

**Potential issues:**
- `.replace({2: 0})` runs **after** NaN replacement, so it only affects remaining valid "2" values. ✓
- `SEXO` semantic: `0 = male`, `1 = female` — consistent across the pipeline. ✓
- This modifies `model_df` **in-place** — if the raw data is needed later, this could cause confusion. Consider working on a `.copy()`.

---

### Q34. Why does the PR curve include `axhline(baseline_rate)`?

```python
baseline_rate = y_test.mean()
ax2.axhline(baseline_rate, color="gray", linestyle="--",
            label=f"Prevalence baseline ({baseline_rate:.3f})")
```

**Expected Answer:**
`y_test.mean()` = 0.236 — the hospitalization prevalence in the test set.

In a Precision-Recall curve, a **random classifier** produces a flat horizontal line at **Precision = prevalence rate**. Drawing this line shows the visual floor: **any curve above it is better than random**. The higher and further right our PR curve sits above this baseline, the better the model discriminates.

---

### Q35. What is `predict_proba(X_test)[:, 1]` and why the `[:, 1]`?

```python
y_test_proba = best_model.predict_proba(X_test)[:, 1]
```

**Expected Answer:**
`predict_proba()` returns a 2D array of shape `(n_samples, n_classes)`:
- Column **0**: P(not hospitalized)
- Column **1**: P(hospitalized)

`[:, 1]` selects **all rows, column index 1** — the probability of hospitalization per patient.

This score is used to:
1. Plot the Precision-Recall curve
2. Compute `roc_auc_score` and `average_precision_score`
3. Drive the operational triage threshold (p̂ ≥ 0.50 → high-risk track)

Both columns sum to 1.0 per patient (complementary probabilities).

---

## 📌 Quick Reference — Key Numbers to Memorize

| Metric | Value |
|---|---|
| Total patients | 263,007 |
| Training set | 210,405 (80%) |
| Test set | 52,602 (20%) |
| Class imbalance | 3.2:1 (76.4% outpatient / 23.6% hospitalized) |
| Dummy PR-AUC | 0.2364 |
| Best model PR-AUC | **0.5771** (+132% over baseline) |
| Best model ROC-AUC | 0.8044 |
| Recall (sensitivity) | **70.1%** |
| Precision | 45.8% |
| Age importance | 55.5% |
| Young adult FN rate | **82.0%** |
| Zero comorbidity FN rate | **53.9%** |
| Best hyperparameters | n_estimators=150, max_depth=15, min_samples_leaf=10 |
| CV folds | 5-fold Stratified |

---

*35 Questions — 9 Sections — Covering: Data Engineering, Class Imbalance, Preprocessing, Models, Evaluation Metrics, Subgroup Fairness, Code Design, and Critical Extensions.*
