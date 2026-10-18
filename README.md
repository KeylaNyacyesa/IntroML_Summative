# Predicting Who Makes It into Developer Jobs: A Gender Gap Audit

**Research question:** *Do models that predict who has made it into a developer job recognise women, and especially young women, as reliably as men?*

**Mission:** to empower young women in Africa to pursue STEM through inclusive pathways that support skills development, mentorship, and access to educational and career opportunities.

## Dataset
**freeCodeCamp, 2018 New Coder Survey.** This is a real, anonymous online survey of people who had been coding for less than five years, collected August–December 2018.
- **Size:** 31,226 responses × 136 questions.
- **Licence:** Open Database License (ODbL) 1.0.
- **Source:** <https://github.com/freeCodeCamp/2018-new-coder-survey>. The official results article (Q. Larson, freeCodeCamp News, 20 Dec 2018) reports the same headline figures as the file.
- **Target:** *"Are you already working as a software developer?"* 16.5% answered yes.
- **Gender:** 6,259 women and 23,486 men in the modelling set.

## Key results

| | Value |
|---|---|
| Best traditional model | E6 histogram gradient boosting (defaults): test PR-AUC **0.654**, ROC-AUC **0.887** |
| Best deep model | E12 Keras Functional API + country embedding + dropout/L2/batch norm: test PR-AUC **0.629** |
| E6 − E12 | +0.025, bootstrap 95% CI [+0.008, +0.042] |
| Recall among real developers, women vs men | **63.4% vs 69.5%** (gap −0.060, 95% CI [−0.100, −0.023]) |
| Stability over 5 random splits | women − men gap mean **−0.065**, significant in **5/5** splits; young women − young men mean **−0.048**, significant in **1/5**: same direction, not robust |
| Best mitigation | E17 group-specific threshold: gap −0.029 (CI includes 0), no loss of PR-AUC |

## Repository contents
```
fcc_summative.ipynb        main notebook (executed; all outputs included)
requirements.txt           dependencies
requirements_frozen.txt    exact versions used for the reported run
experiment_table.csv       every run: settings, metrics, one-line observation
experiments_log.csv        raw per-run log written by the notebook
hgb_search_results.csv     all 20 hyper-parameter configurations (E7 search)
fairness_*.csv             fairness audit tables (out-of-fold and test set)
figures/                   all figures used in the report
experiment_logs.zip        TensorBoard logs (logs/summary/<run id>; Keras per-epoch logs in logs/<run id>)
wandb_offline_runs.zip     Weights & Biases offline runs
```

## How to reproduce
**Google Colab:** open the notebook and run *Runtime → Run all*. The notebook mounts Drive and reads `MyDrive/fcc/2018-new-coder-survey.csv` if present; otherwise it downloads the file.

**Locally:**
```bash
pip install -r requirements.txt
pip install -r requirements_frozen.txt    # optional: the exact Colab versions
jupyter nbconvert --to notebook --execute fcc_summative.ipynb      # about 15 min on a 2-core CPU
```
- Set `QUICK_TEST=1` for a 3-minute dry run.

**View the logs:** `unzip experiment_logs.zip && tensorboard --logdir logs`

## Pipeline
1. **Leakage audit:** 22 skip-logic questions (shown only to developers or only to learners), 5 questions describing the current job, and 12 ID/date/free-text columns are excluded.
2. **Cleaning:** rows without a target are dropped, along with 1 exact duplicate. Impossible ages, hours and experience values are set to missing. Skewed counts are log-transformed.
3. **Features:** 58 inputs in four groups: background, learning intensity, learning resources, and community & mentorship.
4. **Split:** 71/14/14, stratified by target and grouped by internet connection so that no network appears in two partitions.
5. **Experiments, logged to CSV, TensorBoard and W&B offline:**
   - **Traditional ML (E0–E8):** baseline; logistic-regression feature ablation; random forest; gradient boosting with defaults and with grouped randomised search; RBF SVM.
   - **Deep learning (E10–E14d):** Keras Sequential and Functional networks fed by `tf.data`, including a country embedding, dropout/L2/batch normalisation, class weights, and learning-rate/capacity/batch sweeps.
   - **Fairness mitigation (E15–E17):** removing gender, gender × age reweighting, and a group-specific threshold.
6. **Evaluation:**
   - Performance: PR-AUC (selection metric), ROC-AUC, F1, precision and recall, learning curves, confusion matrices, ROC/PR/calibration curves, permutation importance, and the threshold trade-off.
   - Fairness: an out-of-fold audit of all 30,843 respondents covering equal opportunity, disparate impact, ABROCA and mean scores, with bootstrap confidence intervals and an experience-controlled analysis.

## Saved models (Drive folder)
- `best_traditional.joblib`: the full scikit-learn pipeline (E6). Load it with `joblib.load(...)` and call `.predict_proba(df[feature_columns])`. The feature columns and decision threshold are in `traditional_threshold.joblib`.
- `best_deep.keras`, `best_deep.h5` and `best_deep_savedmodel/`: the Keras network (E12). It needs `keras_preprocessing.joblib`, which contains the fitted preprocessor, the country vocabulary and the threshold.

## Limitations
- Respondents are self-selected online volunteers, and the data is from 2018.
- The data is cross-sectional, so the model recognises a profile; it does not forecast who will get a job.
- Gender is recorded only as a binary field.
- Some features may be consequences of having a job.
- Only 323 respondents are African women (39 of them developers), so results for Africa are descriptive only.
