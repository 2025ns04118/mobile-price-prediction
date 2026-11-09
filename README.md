# Mobile Price Prediction — Multi-Class Classification

**S1-26_DSEZG568 Applied Machine Learning**

A comparison of four classifiers — Decision Tree, Random Forest, SVM (RBF) and Gradient
Boosting — plus a heterogeneous soft-voting ensemble, on the **Mobile Price Prediction**
dataset. The goal is to categorise mobile phones into four price ranges (0–3) based on
their technical specifications and deploy the best model with Streamlit.

## Summary

Five models were trained and tuned (5-fold stratified cross-validation) to predict
`price_range` on a perfectly balanced dataset (500 phones per class). **RAM is overwhelmingly
the dominant predictor** (r ≈ 0.92 with the target). The **soft-voting ensemble of Random
Forest + RBF-SVM + Gradient Boosting** wins with **95.5% hold-out accuracy, 0.955 macro-F1
and 0.996 ROC-AUC**. All 18 remaining errors are adjacent-band mistakes — the model never
confuses a budget phone with a flagship.

---

## a. Problem statement

A mobile manufacturing company wants to categorise mobile phones into four price ranges from
product specifications (battery power, RAM, display resolution, connectivity features, etc.).
This is a **multi-class classification** problem: given a phone's specifications, predict
`price_range` ∈ {0 (low cost), 1 (medium cost), 2 (high cost), 3 (very high cost)}.

## b. Dataset description

**Source:** Mobile Price Prediction — provided with the assignment.

| Property | Value |
|---|---|
| Training instances | **2,000** phones |
| Test instances | **1,000** phones (unlabelled — scoring only) |
| Features | **20** (all numeric after encoding) |
| Target | `price_range` — 4 ordered classes (0, 1, 2, 3) |
| Class balance | 500 per class (25% each) — **perfectly balanced** |
| Evaluation split | 1,600 train / 400 validation (80/20, stratified, `random_state=42`) |

### Features

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | battery_power | numeric | Total battery energy (mAh) |
| 2 | blue | categorical (Yes/No) | Bluetooth present |
| 3 | clock_speed | numeric | Processor clock speed (GHz) |
| 4 | dual_sim | categorical (Yes/No) | Dual-SIM support |
| 5 | fc | numeric | Front camera (megapixels) |
| 6 | four_g | binary (0/1) | 4G support |
| 7 | int_memory | numeric | Internal memory (GB) |
| 8 | m_dep | numeric | Mobile depth (cm) |
| 9 | mobile_wt | categorical (Low/Med/High) | Weight band |
| 10 | n_cores | numeric | Number of processor cores |
| 11 | pc | numeric | Primary camera (megapixels) |
| 12 | px_height | numeric | Screen resolution height (pixels) |
| 13 | px_width | numeric | Screen resolution width (pixels) |
| 14 | ram | numeric | Random access memory (MB) |
| 15 | sc_h | numeric | Screen height (cm) |
| 16 | sc_w | numeric | Screen width (cm) |
| 17 | talk_time | numeric | Battery talk-time (hours) |
| 18 | three_g | binary (0/1) | 3G support |
| 19 | touch_screen | categorical (Yes/No) | Touch screen |
| 20 | wifi | categorical (Yes/No) | Wi-Fi support |

### Cleaning and preprocessing

| Step | What was done | Why |
|---|---|---|
| Structural cleanup | Dropped `id` and 3 junk `Unnamed:` columns from the test CSV | Identifiers leak nothing; junk columns are CSV artefacts |
| Encoding | Label encoding for Yes/No columns; ordinal encoding for `mobile_wt` (Low=0, Med=1, High=2) | Models need numeric input; ordinal vs one-hot compared via CV — ordinal wins marginally (0.884 vs 0.878) |
| Invalid values | `px_height=0` and `sc_w=0` median-imputed (training-set medians only) | Physically impossible — a phone cannot have zero pixel height or zero screen width |
| Outliers | 20 flagged points (18 in `fc`, 2 in `px_height`) **retained** | Real premium specs; tree models are order-based, not magnitude-sensitive |
| Skewness | Measured; **no transform applied** | Monotone transforms are invisible to trees; StandardScaler handles SVM |
| Feature engineering | 6 new features: `px_area`, `screen_area`, `ppi`, `total_camera`, `ram_per_core`, `battery_per_talk_time` | Physically meaningful; `ram_per_core` reaches r=0.451; kept because CV shows no degradation |
| Scaling | `StandardScaler` inside `Pipeline` | Mandatory for RBF-SVM; inside pipeline to prevent data leakage across CV folds |
| Feature importance | Gini, mutual information, permutation importance compared | Three lenses guard against single-method bias; all rank `ram` first |

## c. Project structure

```
AML/
├── Group_XX.ipynb                      # Complete notebook (all 9 sections)
├── Group_XX.html                       # HTML export with all outputs
├── app.py                              # Streamlit deployment
├── mobile_price_model.pkl              # Persisted model bundle (joblib)
├── Mobile_Price_Prediction_train.csv   # Training data (2,000 rows)
├── Mobile_Price_Prediction_test.csv    # Test data (1,000 rows, unlabelled)
├── test_predictions.csv                # Predictions on the test set
├── Notebook_Explanation.md             # Cell-by-cell walkthrough for viva
├── DSEZG568 COURSE HANDOUT.docx        # Course handout
├── S1_26_AML_Assignment.pdf            # Assignment problem statement
├── streamlit_app_screenshot.png        # Screenshot of the running app
└── README.md                           # This file
```

## d. Models used

All five models are wrapped in a `Pipeline(StandardScaler → clf)` and tuned with
**`GridSearchCV`** over 5-fold stratified CV. The scaler refits per fold — no leakage.

| Model | Family | Tuned hyper-parameters | Best CV Accuracy |
|---|---|---|---|
| Decision Tree | Single splitter | `max_depth`, `min_samples_leaf`, `criterion` (gini/entropy) | ~82% |
| Random Forest | Bagging ensemble | `n_estimators` [300, 600], `max_depth` [None, 12, 20], `max_features` [sqrt, 0.5], `min_samples_leaf` [1, 2] | ~88% |
| SVM (RBF) | Kernel / margin | `C` [1, 10, 100, 300], `gamma` [scale, 0.01, 0.05] | ~96% |
| Gradient Boosting | Boosting ensemble | `n_estimators` [200, 400], `learning_rate` [0.05, 0.1], `max_depth` [2, 3] | ~92% |
| **Voting Ensemble** | Heterogeneous | RF + SVM + GB, soft voting (probability averaging) | ~95% |

### Why these models?

| Model | Why chosen for this problem |
|---|---|
| **Decision Tree** | EDA showed RAM creates near-clean axis-parallel splits. DT discovers these directly. Fully interpretable baseline — a plottable rule set the business can audit. |
| **Random Forest** | A single tree on 2,000 rows overfits (high variance). Bagging 600 decorrelated trees averages that variance away. The 600 estimators were selected by GridSearchCV from [300, 600] — more trees = more averaging = lower variance, with diminishing returns beyond ~600. |
| **SVM (RBF)** | A structurally *different* hypothesis class — smooth curved boundaries in kernel space instead of axis-parallel boxes. With 2,000 samples and 22 features, SVM is cheap and margin maximisation generalises well. The natural counterweight to the tree family. |
| **Gradient Boosting** | Builds trees sequentially, each correcting the previous ensemble's residual errors. Where bagging (RF) attacks *variance*, boosting attacks *bias* — sharpening the hard adjacent-class boundaries (class 1 vs 2) that EDA flagged as overlapping. Shallow trees (`max_depth` 2–3) as weak learners; power comes from sequential correction. |
| **Voting Ensemble** | Averages predicted probabilities from RF + SVM + GB. Three structurally different models make different kinds of errors, so averaging partly cancels each. DT excluded — too weak (86.5%) and too correlated with RF (both tree-based). |

### Hold-out test results (n = 400)

| Model | Val Accuracy | Macro-F1 | ROC-AUC | Train–Val Gap | Fit Diagnosis |
|---|---|---|---|---|---|
| Decision Tree | 86.5% | ~0.865 | — | Peaks early then falls | **Overfitting** (pruned by GridSearch) |
| Random Forest | 93.5% | ~0.935 | — | 6.4 pts | Mild overfitting |
| SVM (RBF) | 92.8% | ~0.928 | — | 5.8 pts | Mild overfitting |
| Gradient Boosting | 92.0% | ~0.920 | — | 7.7 pts | Mild overfitting |
| **Voting Ensemble** | **95.5%** | **0.955** | **0.996** | **4.4 pts** | **Just right** |

### Why accuracy is valid here

The dataset is **perfectly balanced** (25% per class), so accuracy is directly interpretable
as "fraction of phones priced into the right band." Macro-F1 (0.9552) tracking accuracy
(0.9550) almost exactly confirms no class is being sacrificed. On an imbalanced dataset,
accuracy would be worthless — but that is not the case here.

### Error analysis

The confusion matrix shows **all 18 remaining errors are adjacent-band mistakes** (e.g.
class 1 confused with class 2). **Zero errors jump two or more bands** — the model never
confuses a budget phone with a flagship. The weakest band is class 1 "medium" (recall 0.96,
precision 0.91): the model occasionally promotes a low-cost phone into the medium band,
exactly where the RAM distributions overlap.

### Overfitting / underfitting diagnosis

- **Decision Tree**: Validation curve shows training accuracy → 1.0 while CV score peaks then
  falls — textbook variance explosion. GridSearch selected a shallow, pruned tree to control this.
- **RF, SVM, GB**: Mild overfitting (gaps 5.8–7.7 pts). Training accuracy is near-perfect
  while hold-out sits in the low 90s. Tolerable, but a concrete reason to deploy the ensemble.
- **Voting Ensemble**: Gap of only 4.4 pts — the only "just right" high-accuracy model.
  Averaging three differently-biased members damps each one's variance.
- **Learning curve**: Flattened by ~1,600 samples. The model is not starved of data; remaining
  error is irreducible overlap between adjacent price bands.

## e. Key EDA findings

1. **`ram` is the dominant driver** (r ≈ 0.92) — four nearly disjoint RAM bands, one per class.
2. **`battery_power`** (r ≈ 0.20), **`px_width`** (r ≈ 0.17), **`px_height`** (r ≈ 0.15) are
   weak but consistently positive secondary predictors.
3. Everything else (connectivity flags, cameras, weight, depth) has |r| < 0.05 — nearly uninformative on its own.
4. **RAM is monotonic but not linear** — slightly S-shaped, with growing within-class spread
   for middle classes. Piecewise/threshold learners (trees) model this naturally.
5. **Interactions visible**: in the RAM–battery scatter, class boundaries tilt with battery
   power — a phone with borderline RAM gets pushed up by a larger battery. Favours tree
   ensembles and RBF-SVM over linear models.

## f. Streamlit deployment

The model is persisted as `mobile_price_model.pkl` (via joblib) containing:
- Fitted pipeline (StandardScaler + VotingClassifier)
- Feature order, binary column maps, ordinal encoding maps
- Imputation medians (training-set only)
- Model metrics for display

### App capabilities

| Feature | Description |
|---|---|
| **Single prediction** | Sidebar form for every raw specification → predicted price band + confidence + full probability distribution |
| **Batch prediction** | Upload a CSV → scored table with download button |
| **Model insight** | Metrics of all 5 models compared + permutation importance ranking |

### Running the app

```bash
pip install streamlit scikit-learn pandas numpy joblib
streamlit run app.py
```

The app opens at `http://localhost:8501`. `app.py` and `mobile_price_model.pkl` must be in the
same folder.

## g. Novelty and innovation

1. **Evidence-based decisions** — ordinal vs one-hot encoding, keep vs drop engineered features,
   no-skew-transform: each settled by cross-validated comparison, not assumption.
2. **Data-quality bug found and fixed** — `px_height=0` and `sc_w=0` are physically impossible;
   median-imputed with training-set-only statistics.
3. **Controlled class-imbalance experiment** — artificially skewed copy tested with
   `class_weight`, random over-sampling, SMOTE, SMOTE-Tomek; quantified effect on macro-F1
   and per-class recall.
4. **Three-lens feature importance** — Gini + mutual information + permutation importance
   combined into a consensus ranking.
5. **Heterogeneous soft-voting ensemble** — bagging + boosting + kernel model, chosen for error
   diversity, not three variants of the same learner.
6. **Graphical overfitting diagnosis** — learning curve + validation curve make bias–variance
   trade-off visible.
7. **Ordinal error analysis** — errors split into adjacent-band and 2+-band mistakes; all 18
   errors are adjacent-band.
8. **Production-grade Streamlit UI** — single/batch/insight tabs, not a bare form.
9. **Deployment verified** — artefact reloaded from disk, re-queried, predictions asserted to match.

## h. Reproducing the results

Everything is seeded with `RANDOM_STATE = 42`, so all splits, tuned parameters and metrics
reproduce exactly.

```bash
# Run the notebook end-to-end
jupyter notebook Group_XX.ipynb

# Or just run the Streamlit app
streamlit run app.py
```

## References

1. Mobile Price Prediction dataset — provided with the DSEZG568 assignment.
2. scikit-learn: Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011.
3. DSEZG568 Applied Machine Learning — course handout.
