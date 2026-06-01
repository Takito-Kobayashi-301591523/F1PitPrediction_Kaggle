# F1 Pit Stop Prediction: Complete Reflection Report

This report is intentionally long and detailed. It is designed as a GitHub retrospective that I can reread later and still understand what happened, what worked, what failed, and what I should do differently in future Kaggle competitions.

The report is written first in English and then in Japanese.

---

## English Report

### 0. Coverage Checklist

This version includes the requested items: competition objective, terminology, data overview including original data, why experiments were run locally, column meanings, EDA with images, experiment log, score improvements and score drops, good/bad features with numbers where available, interpretation, final model details, GitHub structure, top-notebook difference analysis, RealMLP reflection, blending/rank blending lessons, and generalized Kaggle learnings.

### 1. Competition Objective

The goal of the Kaggle competition was to predict `PitNextLap`, a binary target that indicates whether a Formula 1 car will enter the pit lane on the next lap. Each row represents lap-level information for a driver in a race or session. The model receives information such as driver, race, year, tyre compound, stint, tyre age, lap number, race progress, position, lap time, lap-time delta, and degradation-related values.

This is a **binary classification** problem. For each `id` in the test set, the submission must output a probability-like value for `PitNextLap`. Because the evaluation metric is ROC AUC, the model does not only need to output calibrated probabilities. It mainly needs to rank rows well: rows that actually become `PitNextLap = 1` should generally receive higher scores than rows that remain `PitNextLap = 0`.

### 2. Complete Glossary of Important Terms

| Term | Explanation | Concrete example in this project |
|---|---|---|
| Binary classification | A prediction problem with two possible classes. | `PitNextLap` is either 0 or 1. |
| Target / objective variable | The column the model tries to predict. | `PitNextLap`. |
| Feature / explanatory variable | A column used as model input. | `TyreLife`, `LapNumber`, `Compound`. |
| Raw feature | A feature already included in the CSV. | `RaceProgress`. |
| Engineered feature | A new feature created from existing columns. | `Race_Compound_Stint`, tyre-life ratio, rolling mean. |
| Train data | Data with target values used for fitting the model. | `train.csv`. |
| Test data | Data without target values; predictions are submitted for it. | `test.csv`. |
| Sample submission | Template showing the required submission format. | `sample_submission.csv`. |
| Original data | External/source F1 strategy data separate from the Playground train/test files. | Useful as optional extra data or reference, but must be checked for distribution mismatch. |
| Synthetic data | Data generated or transformed by the competition host rather than directly collected as the original raw source. | Playground competitions often provide synthetic train/test data. |
| ROC AUC | Ranking metric: how often positives are ranked above negatives. | Higher AUC means pit-next-lap rows are ranked higher. |
| OOF prediction | Out-of-fold prediction: each train row is predicted by a model that did not train on that row. | Used to compare LightGBM, CatBoost, XGBoost, and blending weights. |
| Fold | One split in cross-validation. | In 5-fold CV, each row is validation once. |
| Cross-validation | Repeated train/validation splitting to estimate generalization. | Used to avoid judging only by training score. |
| StratifiedKFold | CV split preserving target ratio in each fold. | Useful for imbalanced binary targets. |
| Group CV | CV that keeps related groups together to reduce leakage. | Could group by race/session if leakage risk is high. |
| Leakage | Accidentally using information unavailable at prediction time. | A feature that directly tells the future pit stop would be leakage. |
| Feature engineering | Creating better inputs from existing data. | `Race_Year`, rolling lap-time mean, frequency encoding. |
| Frequency encoding | Replacing/adding how often a category appears. | `Driver_count` or `Race_Compound_Stint_freq`. |
| Target encoding | Encoding a category using the target mean calculated in a leakage-safe CV way. | Historical pit rate for `Race_Compound`, computed fold-safely. |
| Rolling feature | Statistic over recent rows in a sequence. | 3-lap rolling mean of `LapTime_Delta`. |
| Lag feature | Previous row's value within a sequence. | Previous lap's `LapTime_Delta` for the same driver/race/stint. |
| Group statistics | Mean/std/min/max by group. | Mean `LapTime_Delta` by `Race_Year`. |
| Standard deviation | Measures how spread out values are inside a group. | If lap times in a race have high std, a moderate delta may not be abnormal. |
| Z-score / deviation | How far a value is from its group average. | Current lap is 1.5 std slower than other laps in the same race. |
| Pruning | Removing features judged unimportant. | Importance-based pruning to 150/180 features worsened performance. |
| Ablation / AB test | Add/remove one feature group and compare validation score. | AB-selected features improved LightGBM from 0.955756 to 0.958394. |
| Ensemble | Combining multiple models. | LightGBM + CatBoost. |
| Blending | Weighted averaging of predictions. | `0.427 * LGBM + 0.573 * CatBoost`. |
| Rank blending | Averaging ranks rather than raw probabilities. | Useful for AUC because AUC cares about order. |
| Stacking | Training a second-level model on OOF predictions. | Logistic regression stacker over model OOF predictions. |
| Hyperparameter | A setting chosen before/during training. | `learning_rate`, `max_depth`, `num_leaves`. |
| Optuna | A tool that searches hyperparameters automatically. | Try many learning rates/depths and keep the best CV AUC. |
| Early stopping | Stop training when validation score no longer improves. | Prevents too many trees/epochs from overfitting. |
| Overfitting | Model memorizes validation-like patterns rather than general rules. | Too many noisy features can raise CV but hurt private LB. |
| Public / Private LB | Kaggle public/private leaderboard splits. | Public score can differ from final private score. |
| LightGBM | Fast gradient boosting tree model. | Strong tabular baseline. |
| CatBoost | Gradient boosting model with strong categorical handling. | Strongest single model in the main path. |
| XGBoost | Classic optimized gradient boosting model. | Tested, but weak in this project. |
| RealMLP | Improved MLP for tabular data. | Not used by me, but top notebooks used/blended it. |

### 3. Data Overview

#### 3.1 Main competition files

| dataset           |   rows |   columns |   missing_cells |   missing_pct |   duplicate_rows |
|:------------------|-------:|----------:|----------------:|--------------:|-----------------:|
| train             | 439140 |        16 |               0 |             0 |                0 |
| test              | 188165 |        15 |               0 |             0 |                0 |
| sample_submission | 188165 |         2 |               0 |             0 |                0 |

The training data has 439,140 rows and 16 columns including the target. The test data has 188,165 rows and 15 columns. The target is imbalanced but not extremely rare: `PitNextLap = 1` accounts for about 19.90% of the training data.

This wording matters. Saying the event is “rare” would be misleading. It is not 1% or 0.1%. It is simply not balanced 50:50. Since the models used here are gradient boosting models and the metric is AUC, this level of imbalance is manageable.

#### 3.2 Original data

In addition to the Kaggle Playground `train.csv` and `test.csv`, there was an **original F1 strategy dataset** used as a reference or optional external/additional data source. This original data is different from the competition's generated/synthetic train-test files. It can be useful because it may contain more natural strategy patterns, but it can also have different distributions, naming conventions, missing columns, or target definitions.

Therefore, original data should not be blindly merged. The correct process is:

1. Align column names and meanings with the competition data.
2. Remove or transform columns that do not exist at prediction time.
3. Compare distributions against train/test.
4. Run CV with and without original data.
5. Keep it only if OOF and leaderboard behavior are stable.

The raw data itself should not be committed to GitHub. Instead, the repository should include a `data/README.md` explaining where to download the Kaggle competition files and where to place the optional original dataset locally.

### 4. Why I Ran Experiments Locally

Most experiments were run on my local PC instead of directly inside a Kaggle Notebook. The reason was not that Kaggle was impossible, but that local execution made iteration easier.

Local execution helped because I could keep many intermediate artifacts such as OOF predictions, feature importance files, selected feature lists, calibration tables, and submission candidates. I could also organize experiments by folder, rerun scripts repeatedly, and clean the repository later. Kaggle Notebooks are convenient for sharing, but they can be more restrictive when experimenting quickly under time pressure.

The ideal workflow for next time is: prototype locally, save every OOF/test prediction, then create a clean Kaggle Notebook or GitHub report after the modeling path is stable.

### 5. Column Meanings and EDA-Type Summary

| Column                 | Meaning                                                                                                                           | Type                          | Modeling note                                                                              |
|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------|:------------------------------|:-------------------------------------------------------------------------------------------|
| id                     | Unique row identifier. It is useful only for joining predictions and should not be used as a predictive feature.                  | ID                            | Do not train on this column.                                                               |
| Driver                 | Driver code or anonymized driver ID. It may capture driver/team/strategy style, but high cardinality requires careful validation. | Categorical, high cardinality | Frequency encoding or CatBoost-style categorical handling can help.                        |
| Compound               | Tyre compound such as HARD, MEDIUM, SOFT, INTERMEDIATE, WET.                                                                      | Categorical                   | Strong strategic context; HARD showed a higher pit-next-lap rate than MEDIUM in EDA.       |
| Race                   | Grand Prix or session name. Race conditions, lap length, and strategy windows differ by race.                                     | Categorical                   | Race-specific features and Race × Year interactions can be useful.                         |
| Year                   | Season year.                                                                                                                      | Ordinal/discrete numeric      | Strong distribution differences appeared, so it must be used carefully.                    |
| PitStop                | Flag related to pit stop information at the current row.                                                                          | Binary/discrete               | Useful but should be checked for leakage-like behavior.                                    |
| LapNumber              | Current lap number.                                                                                                               | Ordinal numeric               | Pit probability rises around strategic windows; one of the strongest numeric correlations. |
| Stint                  | Current tyre stint number.                                                                                                        | Ordinal numeric               | Stint 2 had a high pit-next-lap rate; interacts with tyre age and race progress.           |
| TyreLife               | Number of laps on the current tyre.                                                                                               | Numeric                       | Strongest numeric relationship with the target in this EDA.                                |
| Position               | Current race position.                                                                                                            | Ordinal numeric               | Weak linear correlation but can matter through strategic context.                          |
| LapTime (s)            | Lap time in seconds.                                                                                                              | Continuous numeric            | Contains large outliers; raw value alone is less informative than race-relative values.    |
| LapTime_Delta          | Difference from a reference lap time.                                                                                             | Continuous numeric            | Outliers exist; relative/running features may be more useful than raw correlation.         |
| Cumulative_Degradation | Accumulated degradation or pace-loss signal.                                                                                      | Continuous numeric            | Negative correlation with target; needs clipping/relative features.                        |
| RaceProgress           | Fraction of race/session completed.                                                                                               | Continuous numeric in [0,1]   | Useful for identifying strategic phase; interacts with stint and tyre life.                |
| Position_Change        | Change in track/race position.                                                                                                    | Discrete numeric              | Weak correlation alone, but can identify unstable race situations.                         |
| PitNextLap             | Target. 1 means the car pits on the next lap.                                                                                     | Binary target                 | Predicted as probability for AUC evaluation.                                               |

#### Numeric/statistical summary from train

| column                 | role    | dtype   |   missing |   unique | min        | median      | mean        | max         |
|:-----------------------|:--------|:--------|----------:|---------:|:-----------|:------------|:------------|:------------|
| id                     | id      | int64   |         0 |   439140 | 0.0000     | 219569.5000 | 219569.5000 | 439139.0000 |
| Driver                 | feature | str     |         0 |      887 |            |             |             |             |
| Compound               | feature | str     |         0 |        5 |            |             |             |             |
| Race                   | feature | str     |         0 |       26 |            |             |             |             |
| Year                   | feature | int64   |         0 |        4 | 2022.0000  | 2024.0000   | 2023.5235   | 2025.0000   |
| PitStop                | feature | int64   |         0 |        2 | 0.0000     | 0.0000      | 0.1361      | 1.0000      |
| LapNumber              | feature | int64   |         0 |       78 | 1.0000     | 19.0000     | 23.1059     | 78.0000     |
| Stint                  | feature | int64   |         0 |        8 | 1.0000     | 2.0000      | 1.7891      | 8.0000      |
| TyreLife               | feature | float64 |         0 |       78 | 1.0000     | 12.0000     | 14.1582     | 77.0000     |
| Position               | feature | int64   |         0 |       20 | 1.0000     | 10.0000     | 9.6303      | 20.0000     |
| LapTime (s)            | feature | float64 |         0 |    37719 | 67.6940    | 90.5210     | 90.9487     | 2507.6070   |
| LapTime_Delta          | feature | float64 |         0 |    57532 | -2403.8950 | -0.2950     | -3.7700     | 2423.9320   |
| Cumulative_Degradation | feature | float64 |         0 |   142701 | -274.5640  | -20.9940    | -25.7218    | 2412.0260   |
| RaceProgress           | feature | float64 |         0 |     1898 | 0.0128     | 0.2692      | 0.3377      | 1.0000      |
| Position_Change        | feature | float64 |         0 |       37 | -18.0000   | 0.0000      | 0.1015      | 18.0000     |
| PitNextLap             | target  | float64 |         0 |        2 | 0.0000     | 0.0000      | 0.1990      | 1.0000      |

### 6. EDA: What Was Visualized and What It Means

The EDA was designed to cover every non-`id` column. For numeric columns, the report checks histograms, box plots, target-split distributions, class-wise averages, correlation with the target, and binned pit rates when appropriate. For categorical columns, it checks bar charts, category counts, category-wise pit rates, top categories, and compact share views. For ordered/time-like columns, it emphasizes trends such as pit rate by `LapNumber`, `TyreLife`, `RaceProgress`, and `Year`.

#### 6.1 Correlation with target

| column                 |   corr_with_target |   abs_corr_with_target |
|:-----------------------|-------------------:|-----------------------:|
| TyreLife               |             0.2735 |                 0.2735 |
| LapNumber              |             0.2671 |                 0.2671 |
| Stint                  |             0.1982 |                 0.1982 |
| RaceProgress           |             0.1855 |                 0.1855 |
| Cumulative_Degradation |            -0.1674 |                 0.1674 |
| Year                   |             0.1253 |                 0.1253 |
| PitStop                |             0.0486 |                 0.0486 |
| Position_Change        |             0.0462 |                 0.0462 |
| LapTime (s)            |            -0.0341 |                 0.0341 |
| Position               |             0.0213 |                 0.0213 |
| LapTime_Delta          |            -0.0049 |                 0.0049 |

#### 6.2 Category-wise count and pit rate examples

| column   | category                  |   count |   pit_rate |
|:---------|:--------------------------|--------:|-----------:|
| Compound | HARD                      |  170518 |     0.3275 |
| Compound | SOFT                      |   38744 |     0.1935 |
| Compound | INTERMEDIATE              |   17382 |     0.1523 |
| Compound | MEDIUM                    |  211141 |     0.1011 |
| Compound | WET                       |    1355 |     0.0251 |
| Race     | Chinese Grand Prix        |    7311 |     0.3886 |
| Race     | Monaco Grand Prix         |   21539 |     0.3574 |
| Race     | Spanish Grand Prix        |   20483 |     0.32   |
| Race     | Bahrain Grand Prix        |   19535 |     0.2875 |
| Race     | Belgian Grand Prix        |    9002 |     0.2804 |
| Race     | Emilia Romagna Grand Prix |   15483 |     0.2726 |
| Race     | São Paulo Grand Prix      |   11497 |     0.2537 |
| Race     | Hungarian Grand Prix      |   22481 |     0.2393 |
| Race     | Saudi Arabian Grand Prix  |   18111 |     0.2274 |
| Race     | Las Vegas Grand Prix      |   12479 |     0.2253 |
| Stint    | 2.0                       |  129536 |     0.3911 |
| Stint    | 3.0                       |   69238 |     0.2931 |
| Stint    | 4.0                       |   18903 |     0.1717 |
| Stint    | 1.0                       |  216288 |     0.0598 |
| Stint    | 5.0                       |    4281 |     0.053  |
| Stint    | 8.0                       |      50 |     0.02   |
| Stint    | 6.0                       |     728 |     0.0192 |
| Stint    | 7.0                       |     116 |     0      |
| Position | 13.0                      |   24850 |     0.2351 |
| Position | 14.0                      |   23824 |     0.2299 |
| Position | 15.0                      |   23905 |     0.2199 |
| Position | 16.0                      |   21769 |     0.2134 |
| Position | 8.0                       |   24777 |     0.2072 |
| Position | 9.0                       |   24416 |     0.2065 |
| Position | 17.0                      |   19886 |     0.2045 |
| Position | 11.0                      |   25031 |     0.2038 |
| Position | 12.0                      |   24937 |     0.2005 |
| Position | 10.0                      |   24700 |     0.1955 |


#### Target distribution

The target distribution checks how many rows are `PitNextLap = 0` and `PitNextLap = 1`. The positive rate is about 19.90%, so the event is imbalanced but not extremely rare. This is acceptable for gradient boosting models, and AUC is appropriate because it evaluates ranking rather than raw class accuracy.

![Target distribution](reports/figures/eda/target_distribution.png)

#### Numeric correlation heatmap and target correlation

The heatmap shows relationships among numeric columns. This matters because highly related columns can carry overlapping information. The target-correlation plot shows that `TyreLife`, `LapNumber`, `Stint`, and `RaceProgress` are the strongest numeric signals in linear correlation, while `LapTime_Delta` is weak in simple correlation even though it may still matter through nonlinear or group-relative features.

![Numeric correlation heatmap](reports/figures/eda/correlation_heatmap.png)

![Correlation with target](reports/figures/eda/correlation_with_target.png)

#### Per-column visual checks

For numeric columns, each figure is designed to show a histogram, a box plot, the distribution split by `PitNextLap = 0/1`, class-wise averages, and binned pit rates when appropriate. For categorical columns, each figure focuses on category counts, category-wise pit rates, top categories, and a compact share view when useful.

![EDA for Driver](reports/figures/eda/columns/categorical_Driver.png)

![EDA for Compound](reports/figures/eda/columns/categorical_Compound.png)

![EDA for Race](reports/figures/eda/columns/categorical_Race.png)

![EDA for Year](reports/figures/eda/columns/numeric_Year.png)

![EDA for PitStop](reports/figures/eda/columns/numeric_PitStop.png)

![EDA for LapNumber](reports/figures/eda/columns/numeric_LapNumber.png)

![EDA for Stint](reports/figures/eda/columns/numeric_Stint.png)

![EDA for TyreLife](reports/figures/eda/columns/numeric_TyreLife.png)

![EDA for Position](reports/figures/eda/columns/numeric_Position.png)

![EDA for LapTime (s)](reports/figures/eda/columns/numeric_LapTime_s.png)

![EDA for LapTime_Delta](reports/figures/eda/columns/numeric_LapTime_Delta.png)

![EDA for Cumulative_Degradation](reports/figures/eda/columns/numeric_Cumulative_Degradation.png)

![EDA for RaceProgress](reports/figures/eda/columns/numeric_RaceProgress.png)

![EDA for Position_Change](reports/figures/eda/columns/numeric_Position_Change.png)

#### Train/test distribution comparison

These figures compare train and test distributions. Distribution shift matters because a feature can look strong in cross-validation but fail on the leaderboard if train/test have different distributions. This is also why original data must be treated as optional and checked before mixing it into training.

![Distribution comparison: Race](reports/figures/eda/dataset_comparison/compare_Race.png)

![Distribution comparison: Compound](reports/figures/eda/dataset_comparison/compare_Compound.png)

![Distribution comparison: Year](reports/figures/eda/dataset_comparison/compare_Year.png)

![Distribution comparison: LapNumber](reports/figures/eda/dataset_comparison/compare_LapNumber.png)

![Distribution comparison: TyreLife](reports/figures/eda/dataset_comparison/compare_TyreLife.png)

![Distribution comparison: RaceProgress](reports/figures/eda/dataset_comparison/compare_RaceProgress.png)

![Distribution comparison: LapTime (s)](reports/figures/eda/dataset_comparison/compare_LapTime_s.png)

![Distribution comparison: LapTime Delta](reports/figures/eda/dataset_comparison/compare_LapTime_Delta.png)

![Distribution comparison: Cumulative Degradation](reports/figures/eda/dataset_comparison/compare_Cumulative_Degradation.png)

![Distribution comparison: Position](reports/figures/eda/dataset_comparison/compare_Position.png)

![Distribution comparison: Position Change](reports/figures/eda/dataset_comparison/compare_Position_Change.png)

![Distribution comparison: Stint](reports/figures/eda/dataset_comparison/compare_Stint.png)

![Distribution comparison: PitStop](reports/figures/eda/dataset_comparison/compare_PitStop.png)

![Distribution comparison: Driver](reports/figures/eda/dataset_comparison/compare_Driver.png)


#### 6.3 Column-by-column interpretation

| Column | Main EDA interpretation | Why it matters for modeling |
|---|---|---|
| Driver | High-cardinality categorical column with 887 values. Some drivers have much higher/lower pit rates, but rare drivers can be noisy. | Use frequency encoding and possibly CatBoost categorical handling. Driver frequency can proxy whether a driver appears often enough for stable strategy patterns. |
| Compound | `HARD` has much higher pit-next-lap rate than `MEDIUM`; `WET` is rare and low in this data. | Tyre compound directly changes strategy windows. Must interact with stint and race. |
| Race | Pit rates vary widely by race, from low Mexico City/Miami/US to high Chinese/Monaco/Spanish. | Race context is crucial; race-only and race-combination features are valuable. |
| Year | 2023 has a very different target rate from other years. | Could be real/generated distribution shift; needs careful validation. |
| PitStop | PitStop=1 has higher next-lap pit rate than PitStop=0. | Useful but must be checked for leakage-like behavior. |
| LapNumber | Strong positive relationship with target. Pit probability rises around race strategy windows. | One of the strongest raw numeric features. |
| Stint | Stint 2 has a high pit-next-lap rate. | Represents strategy phase; should interact with tyre life and race progress. |
| TyreLife | Strongest numeric target correlation in EDA. Positive rows have higher average tyre life. | Directly matches the idea that old tyres are more likely to be changed. |
| Position | Weak correlation alone but mid-pack positions show different pit rates. | Strategy can depend on traffic and undercut/overcut possibilities. |
| LapTime (s) | Large outliers exist; raw correlation is negative and modest. | Use clipping, outlier flags, and race-relative values. |
| LapTime_Delta | Simple correlation is weak despite many outliers. | Better as lag/rolling/group-relative features than raw value. |
| Cumulative_Degradation | Negative correlation with target and large outliers. | Useful but should be robustly transformed. |
| RaceProgress | Positive relationship with target. | Captures race phase and should interact with tyre life/stint. |
| Position_Change | Weak alone, but positive rows have higher average position change. | Can capture unstable race situations or strategic reactions. |

### 7. What I Did

| Step | What I did | Why | Result |
|---|---|---|---|
| 1 | Built an initial LightGBM pipeline. | Fast and reliable baseline for tabular data. | Established train/CV/submission loop. |
| 2 | Asked AI to generate 100+ features and AB-tested them. | Wanted to quickly explore many feature ideas. | The approach stayed in the 0.93 range and was discarded. |
| 3 | Rebuilt from problem understanding. | The first broad AI approach lacked F1 strategy logic. | Produced a more stable Stage-2 baseline. |
| 4 | Added safe numeric/domain features. | Tyre age, lap number, race progress, and degradation are related to pit decisions. | Early score improved to 0.944248. |
| 5 | Tried outlier flags. | Extreme lap/degradation values might indicate pit windows. | Slight or unstable gain only. |
| 6 | Tried broad feature expansion. | Race/year/context interactions may add useful signal. | Improved modestly to 0.944658 in early path. |
| 7 | Tried importance-based pruning. | Reduce noise by dropping weak features. | Worsened by about 0.0010–0.0011. |
| 8 | Created a cleaner Stage-2 LightGBM baseline. | Reset to a stronger, compact feature set. | OOF AUC 0.955756. |
| 9 | Ran AB-selected LightGBM features. | Keep only features that improved validation. | OOF AUC 0.958394. |
| 10 | Trained CatBoost on selected features. | CatBoost handles categorical/rule-like structure differently. | OOF AUC 0.959197. |
| 11 | Blended LightGBM and CatBoost. | Reduce model-specific ranking errors. | Best OOF AUC 0.960446. |
| 12 | Added lightweight XGBoost. | Test third-model diversity. | Too weak; final weight 0.0. |
| 13 | Tried CatBoost-2000 follow-up. | Longer CatBoost training might improve. | Strong but below best blend. |

### 8. What Improved or Worsened the Score

| Experiment                            | OOF AUC   | Score impact                           | Interpretation                                                                                                 |
|:--------------------------------------|:----------|:---------------------------------------|:---------------------------------------------------------------------------------------------------------------|
| Early safe numeric feature baseline   | ~0.937000 | starting point                         | Initial broad baseline before more stable engineering.                                                         |
| Safe numeric features                 | 0.944248  | +~0.007248                             | Tyre age, race phase, lap timing and degradation were close to the real pit decision mechanism.                |
| Outlier flags                         | 0.943862  | roughly flat / slight gain in one path | Outlier flags captured abnormal laps but were sparse and unstable.                                             |
| Full feature expansion                | 0.944658  | +0.000410 vs safe numeric path         | Race/year/context features added information, but the broad set also added noise.                              |
| Stage-2 LightGBM baseline             | 0.955756  | new stronger baseline                  | A compact, cleaner feature set with better CV became much stronger than the first large AI feature attempt.    |
| LightGBM AB-selected features         | 0.958394  | +0.002638 vs Stage-2 LGBM              | Direct AB testing kept features that improved validation and removed noisy ideas.                              |
| CatBoost on selected features         | 0.959197  | +0.000803 vs AB-selected LGBM          | Different categorical/rule treatment made it the strongest single model in the main path.                      |
| LightGBM + CatBoost probability blend | 0.960446  | +0.001249 vs CatBoost single           | Two strong models made similar but not identical errors; averaging improved AUC ranking.                       |
| XGBoost lightweight model             | 0.942677  | too low to help                        | The final ensemble assigned XGBoost weight 0.0 because its predictions added more noise than useful diversity. |
| Three-model ensemble                  | 0.960446  | same as 2-model blend                  | Because XGBoost received weight 0.0, this effectively became the LightGBM/CatBoost blend.                      |
| CatBoost-2000 follow-up               | 0.960022  | -0.000424 vs best blend                | Strong standalone run, but did not beat the two-model ensemble.                                                |

The important correction is that **OOF itself is not low or high**. OOF means the prediction method. The metric calculated from OOF predictions, such as **OOF AUC**, is what becomes high or low. Therefore, the correct wording is “OOF AUC was low,” not “OOF was low.”

### 9. Good Features, Bad Features, Numbers, and Reasons

| Source                 | Feature or method                                     | Judgment             | Evidence / number                                                      | Reason                                                                                                              |
|:-----------------------|:------------------------------------------------------|:---------------------|:-----------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|
| Manual / domain        | TyreLife, LapNumber, RaceProgress, Stint interactions | Good                 | TyreLife corr 0.2735; LapNumber corr 0.2671; Stage-2 baseline 0.955756 | Directly connected to pit strategy timing.                                                                          |
| Manual / domain        | Race phase and tyre-age ratios                        | Good                 | Safe numeric path improved from ~0.937 to 0.944248                     | Pit decisions depend on how old the tyres are relative to the race stage.                                           |
| Manual / domain        | Outlier flags for extreme lap time/degradation        | Mixed                | 0.943686 to 0.943862 in early experiments                              | They catch abnormal/pit-window cases, but are sparse and can overfit.                                               |
| Manual / validation    | AB-selected feature set                               | Good                 | LightGBM 0.955756 to 0.958394                                          | Testing features one by one reduced noise.                                                                          |
| Manual / ensemble      | LightGBM + CatBoost blend                             | Very good            | 0.959197 to 0.960446                                                   | Blending reduced model-specific ranking errors.                                                                     |
| AI-generated / broad   | 100+ automatically generated features                 | Bad in first attempt | Result stayed in the 0.93 range and was discarded                      | Many features were not tied to F1 strategy and added noise.                                                         |
| AI-generated / broad   | Blind large lag/rolling/encoding expansion            | Mixed                | Full expansion reached 0.944658 early, but did not become final        | Some signal existed, but validation and leakage control were not mature enough.                                     |
| AI-generated / pruning | Importance-based pruning to 150/180 features          | Bad                  | about -0.0010 to -0.0011 vs full-feature baseline                      | The feature set was not filled with obviously harmful variables; pruning may remove weak-but-complementary signals. |
| Top-notebook idea      | Frequency encoding                                    | Should have tried    | Not in my final score logs; observed in top notebooks                  | Frequency can tell the model whether a driver/race/category is common, stable, or rare/noisy.                       |
| Top-notebook idea      | Group mean/std/deviation features                     | Should have tried    | Not in my final score logs; important in top notebooks                 | They express whether a lap is fast/slow relative to the same race, year, compound, or driver context.               |
| Top-notebook idea      | RealMLP                                               | Should have tried    | Not in my final model; top notebooks used it/blended it                | Different model family could add diversity beyond tree models.                                                      |

The exact individual AUC contribution for every single feature is not fully available in the saved artifacts, so this report avoids inventing numbers. Where exact values are available, it reports them. Where only group-level experiment logs exist, it reports the group-level impact.

### 10. Final Model: What Actually Improved the Score

The best local path was:

```text
Stage-2 LightGBM baseline: 0.955756
→ AB-selected LightGBM:    0.958394  (+0.002638)
→ CatBoost single model:   0.959197  (+0.000803 vs AB LGBM)
→ LGBM/Cat probability blend: 0.960446  (+0.001249 vs CatBoost)
```

The final model improved because of four things:

1. **Cleaner feature set**: The model stopped relying on too many AI-generated features and used features more connected to F1 pit strategy.
2. **AB feature selection**: Features were not accepted just because they sounded clever; they had to improve OOF AUC.
3. **CatBoost addition**: CatBoost captured categorical/rule-like structure differently from LightGBM.
4. **Probability blending**: The LightGBM and CatBoost predictions were similar but not identical, so averaging them improved the ranking.

XGBoost did not help because its OOF AUC was 0.942677. In a weighted ensemble, a model with much lower OOF can hurt the final ranking, so the optimizer assigned it weight 0.0.

### 11. Model Algorithms, Use Cases, Strengths, Weaknesses, and Hyperparameters

#### 11.1 What is Optuna?

Optuna is an automatic hyperparameter optimization framework. In simple terms, it is a smart trial-and-error helper. Instead of manually trying `learning_rate = 0.03`, then `0.05`, then `0.01`, Optuna runs many experiments and uses previous results to decide which settings to try next. It does not remove the need for human judgment: I still need to choose the search space, CV design, metric, runtime budget, and leakage-safe feature pipeline.

#### 11.2 LightGBM

LightGBM is a gradient boosting decision tree model. Imagine a student making many small yes/no rule trees. The first tree makes rough predictions. The next tree focuses on what the previous trees got wrong. After many trees, the model becomes strong.

Middle-school example: if we want to predict whether someone will buy a drink, one tree may ask “Is it hot today?” Another may ask “Is the person exercising?” Another may ask “Is the drink cheap?” LightGBM combines many such small decision rules.

**Use cases:** large tabular data, many numeric features, fast Kaggle baselines, feature importance checks.

**Strengths:** very fast, strong on tabular data, works well with many features, easy to tune.

**Weaknesses:** categorical variables need careful encoding; can overfit if trees are too complex; may be less naturally suited to high-cardinality categories than CatBoost.

| Hyperparameter                      | Meaning                                            | Optuna?    | Comment                                                                   |
|:------------------------------------|:---------------------------------------------------|:-----------|:--------------------------------------------------------------------------|
| n_estimators / num_boost_round      | Number of boosting trees.                          | Yes        | Usually tuned with early stopping; too many without stopping can overfit. |
| learning_rate                       | How strongly each new tree changes the prediction. | Yes        | Lower values are safer but need more trees.                               |
| num_leaves                          | Maximum number of leaves per tree.                 | Yes        | Controls tree complexity; too high can overfit.                           |
| max_depth                           | Maximum depth of each tree.                        | Yes        | Limits complexity; can be -1 for no limit.                                |
| min_child_samples                   | Minimum data points in a leaf.                     | Yes        | Higher values regularize the model.                                       |
| subsample / bagging_fraction        | Fraction of rows used per tree.                    | Yes        | Adds randomness and reduces overfitting.                                  |
| colsample_bytree / feature_fraction | Fraction of columns used per tree.                 | Yes        | Useful when there are many features.                                      |
| lambda_l1 / lambda_l2               | L1/L2 regularization.                              | Yes        | Penalizes overly complex trees.                                           |
| objective                           | Learning task such as binary classification.       | Usually no | Set by the problem, not a free tuning target.                             |
| metric                              | Validation metric such as AUC.                     | Usually no | Should match competition objective.                                       |
| random_state / seed                 | Random seed.                                       | Usually no | Fix for reproducibility; can average multiple seeds later.                |

#### 11.3 CatBoost

CatBoost is also a gradient boosting tree model, but it is especially strong with categorical variables. This matters because F1 data has columns like `Driver`, `Race`, and `Compound`. CatBoost can often use these columns more naturally than a model that requires manual one-hot encoding or target encoding.

Middle-school example: if `Race = Monaco` behaves differently from `Race = Mexico City`, CatBoost can learn that category difference without forcing me to manually convert every race into many dummy columns.

**Use cases:** tabular data with many categorical columns, high-cardinality categories, competitions where category interactions matter.

**Strengths:** strong categorical handling, robust defaults, often great for Kaggle tabular problems.

**Weaknesses:** can be slower than LightGBM; many categorical interaction features still need validation; large feature sets can look strong in CV if validation is not strict.

| Hyperparameter      | Meaning                                 | Optuna?              | Comment                                                      |
|:--------------------|:----------------------------------------|:---------------------|:-------------------------------------------------------------|
| iterations          | Number of boosting iterations/trees.    | Yes                  | Often combined with early stopping.                          |
| learning_rate       | Step size of each tree.                 | Yes                  | Lower values need more iterations.                           |
| depth               | Depth of trees.                         | Yes                  | Main complexity control; too deep can overfit.               |
| l2_leaf_reg         | L2 regularization.                      | Yes                  | Important for avoiding overfit.                              |
| bagging_temperature | Controls Bayesian bootstrap randomness. | Yes                  | Useful for generalization.                                   |
| random_strength     | Randomness in split scoring.            | Yes                  | Can reduce overfitting.                                      |
| border_count        | Number of bins for numeric features.    | Yes                  | Affects numeric feature discretization.                      |
| cat_features        | Which columns are categorical.          | No, selected by data | Must be passed correctly; not usually optimized as a number. |
| loss_function       | Training objective such as Logloss.     | Usually no           | Determined by binary classification task.                    |
| eval_metric         | Validation metric such as AUC.          | Usually no           | Should match competition objective.                          |
| random_seed         | Seed for reproducibility.               | Usually no           | Can run multiple seeds later.                                |

#### 11.4 XGBoost

XGBoost is one of the classic gradient boosting tree models. It is reliable and flexible, and it has been used in many winning tabular solutions. In this project, however, the lightweight XGBoost model was much weaker than LightGBM and CatBoost.

Middle-school example: XGBoost is like another student using similar yes/no rule trees, but with slightly different training rules and regularization. Sometimes that different style helps; here it did not help enough.

**Use cases:** robust tabular baseline, when careful regularization is needed, when comparing multiple tree implementations.

**Strengths:** mature, flexible, strong regularization options.

**Weaknesses:** can be slower; categorical handling may require extra work; if it is much weaker than other models, it should not be forced into the blend.

| Hyperparameter                 | Meaning                                    | Optuna?    | Comment                                |
|:-------------------------------|:-------------------------------------------|:-----------|:---------------------------------------|
| n_estimators / num_boost_round | Number of boosting rounds.                 | Yes        | Tune with early stopping.              |
| eta / learning_rate            | Step size.                                 | Yes        | Lower is safer but slower.             |
| max_depth                      | Tree depth.                                | Yes        | Main complexity control.               |
| min_child_weight               | Minimum sum of instance weight in a child. | Yes        | Higher values regularize.              |
| subsample                      | Row sampling fraction.                     | Yes        | Reduces overfitting.                   |
| colsample_bytree               | Column sampling fraction.                  | Yes        | Useful for many features.              |
| gamma                          | Minimum loss reduction to split.           | Yes        | Higher values make splitting stricter. |
| lambda / alpha                 | L2/L1 regularization.                      | Yes        | Controls complexity.                   |
| objective                      | binary:logistic for binary probability.    | Usually no | Set by task.                           |
| eval_metric                    | auc for validation.                        | Usually no | Should match competition.              |
| tree_method                    | hist/gpu_hist etc.                         | Sometimes  | Often chosen by environment.           |

#### 11.5 RealMLP

RealMLP is an improved multilayer perceptron for tabular data. A normal MLP is a neural network that learns weighted combinations of inputs through layers of neurons. RealMLP improves this idea for tables using strong preprocessing, tuned defaults, regularization, and training tricks. It is not a tree model, so it can make different types of errors from LightGBM/CatBoost/XGBoost.

Middle-school example: tree models learn by asking many yes/no questions. RealMLP is more like mixing many sliders at once: tyre age, race progress, degradation, and position all flow through connected layers, and the network learns smooth combinations.

**Use cases:** tabular classification/regression when I already have strong tree models and need model diversity; medium-to-large datasets; blending with GBDTs.

**Strengths:** different model family, can learn smooth interactions, useful ensemble diversity, strong pre-tuned defaults in PyTabKit/RealMLP implementations.

**Weaknesses:** more sensitive to preprocessing/scaling, random seeds, training time, GPU/CPU setup, and hyperparameters; harder to interpret than trees; may underperform trees on some pure tabular tasks.

| Hyperparameter          | Meaning                                   | Optuna?                            | Comment                                                                                |
|:------------------------|:------------------------------------------|:-----------------------------------|:---------------------------------------------------------------------------------------|
| hidden layers / width   | Size and shape of the neural network.     | Yes, but defaults are often strong | Controls how much representation capacity the network has.                             |
| learning rate           | Gradient descent step size.               | Yes                                | Too high is unstable; too low is slow.                                                 |
| batch size              | Number of rows per mini-batch.            | Yes                                | Affects speed, memory, and generalization.                                             |
| epochs / max epochs     | Training passes over the data.            | Yes                                | Usually use best-epoch selection or early stopping.                                    |
| weight decay            | Regularization on neural weights.         | Yes                                | Reduces overfitting.                                                                   |
| dropout                 | Randomly disables neurons while training. | Yes                                | Can help generalization but too much hurts.                                            |
| preprocessing / scaling | Robust scaling, clipping, encoding.       | Partly no                          | Usually part of the RealMLP recipe; should not be randomly changed without validation. |
| n_refit / ensemble size | How many refits/models to average.        | Yes if resources allow             | Improves stability at extra cost.                                                      |
| random seed             | Initialization and data order.            | Usually no                         | Use for reproducibility; multiple seeds can be ensembled.                              |

### 12. Manual Features vs AI-Generated Features

The first failed path was important. At the beginning, I asked AI to generate over 100 features, then tried to AB-test them and keep only those that improved accuracy. However, the result stayed in the 0.93 range, so I discarded that entire path and rebuilt the model from understanding the problem.

The lesson is not “never use AI.” AI is useful for brainstorming. The problem was that many generated features were not clearly connected to F1 pit strategy. Good Kaggle features are not just complicated columns. They should explain why the target happens.

**Manual/domain features that made sense:** tyre age, lap number, race progress, stint, compound, race context, degradation, relative lap timing, and selected interactions.

**AI-generated features that were risky:** many broad lag/rolling/encoding/digit/outlier features without a clear hypothesis, especially if they were not checked for leakage and CV stability.

### 13. Difference from Top Notebooks

#### 13.1 Positioning of my model

My final model was a solid tree-based solution. It used LightGBM, CatBoost, and a tested XGBoost addition. It reached a competitive final result, but it was still mainly a gradient-boosting solution. The top notebooks went further in three areas: feature context, model diversity, and final blending.

#### 13.2 What top notebooks did that I did not do enough

| Area | My approach | Top-notebook approach | Why it mattered |
|---|---|---|---|
| Category interactions | Used main categories and some engineered features. | Built many combinations such as `Race_Year`, `Driver_Race`, `Race_Compound_Stint`. | F1 strategy depends on context, not only single columns. |
| Frequency encoding | Not a major final feature. | Used counts/frequencies for driver/race/compound combinations. | Frequency tells whether a pattern is common/stable or rare/noisy. |
| Group statistics | Limited. | Used means/stds/differences by race/year/compound/stint/driver. | Shows whether a lap is unusual within the same context. |
| Standard deviation features | Underused. | Used group std features. | Std shows how variable a race/context is; one lap can be judged relative to that context. |
| Lag/rolling features | Not enough in final path. | Used previous lap and rolling statistics. | Pit decisions follow deterioration trends, not only current values. |
| Target encoding | Not central in final. | Used leakage-aware target-like encodings. | Captures historical pit tendency by category. |
| Feature count | Compact selected features. | One silver-medal CatBoost notebook used about 300 features. | CatBoost + early stopping + validation can tolerate many useful context features. |
| RealMLP | Not used. | Used directly or in blends. | Different model family adds diversity. |
| Blending/rank blending | Used weighted probability blend. | Also used OOF correlation, safe blends, rank-based methods, and sometimes stacking. | AUC rewards ranking, so rank methods can help. |

#### 13.3 Why frequency encoding was smart

Frequency encoding means adding how often a category appears. For example, `Driver_count` tells how often a driver appears in the data. This was smart because a frequently appearing driver may represent a more stable strategy pattern. A driver who appears often in races may have more consistent pit strategy behavior, and the model can trust that pattern more. A very rare driver/race combination may be noisy.

This is why the “frequency” idea was impressive: I had thought about the category itself, but not about the reliability of the category.

#### 13.4 Why standard deviation and group-relative features helped

A raw value like `LapTime_Delta = 3` is not enough. In one race, +3 seconds may be very slow. In another race or weather condition, lap times may vary a lot and +3 may be less special. If we calculate the standard deviation within `Race_Year`, we can know whether the current lap is unusually fast or slow compared with nearby context.

For example:

```text
Race A average delta = 0.5, std = 0.8, current delta = 3.0 → very unusual
Race B average delta = 1.5, std = 3.0, current delta = 3.0 → not very unusual
```

This gives the model a relative signal: not just “what is the lap time?” but “how abnormal is this lap inside this race?”

#### 13.5 Why about 300 features could work in a silver-medal CatBoost notebook

A silver-medal CatBoost notebook reportedly used about 300 features. This does not mean “more features is always better.” It worked because the features were structured around race context, and CatBoost can handle categorical patterns well. Early stopping also prevents the model from continuing to memorize noise after validation performance stops improving.

Also, tree models do not use every feature equally. If many features are weak but not harmful, the model may mostly ignore them. The risk remains: if validation is not strict enough, 300 features can overfit. The reason it worked was likely the combination of meaningful feature design, CatBoost categorical handling, regularization, early stopping, and stable folds.

#### 13.6 RealMLP reflection

I should have tried RealMLP because it is a different model family. LightGBM, CatBoost, and XGBoost are all tree-based. Even if they differ, they often make correlated predictions. RealMLP could capture smoother interactions and neural-network-style representations.

In ensemble terms, the best added model is not always the strongest single model. Sometimes a slightly weaker but less-correlated model improves the final blend. For example, if LightGBM and CatBoost OOF predictions are almost identical, their blend has limited new information. If RealMLP has decent AUC but different errors, it can improve final ranking.

#### 13.7 OOF prediction saving, correlation, and ensemble weights

Top solutions saved OOF predictions and test predictions for every model. This enables analysis like:

```text
model_lgbm_oof.csv
model_catboost_oof.csv
model_realmlp_oof.csv
model_xgb_oof.csv
```

Then I can calculate correlations. Example:

```text
corr(LightGBM, CatBoost) = 0.97
corr(LightGBM, RealMLP)  = 0.92
corr(CatBoost, RealMLP)  = 0.91
```

These numbers are illustrative, but the logic is important. If two models are highly correlated, blending them gives limited new information. If a model has good AUC and lower correlation, it may improve the ensemble more. Then I can search weights using OOF AUC:

```text
final_oof = 0.45 * lgbm_oof + 0.40 * cat_oof + 0.15 * realmlp_oof
```

After finding weights on OOF, I apply the same weights to test predictions.

#### 13.8 Blending and rank blending

Normal blending averages probabilities:

```text
final = 0.7 * model_A_probability + 0.3 * model_B_probability
```

Rank blending averages ranks:

```text
rank_A = rank(model_A_probability)
rank_B = rank(model_B_probability)
final_rank_score = 0.7 * rank_A + 0.3 * rank_B
```

This matters because AUC is based on ordering. If one model's probability scale is strange but its order is good, rank blending can still use it. Top Kaggle competitors often look at previous similar tabular AUC competitions or public notebooks to learn safe blend patterns, but the key is not blindly copying. The correct way is to validate the blend using OOF predictions and avoid overfitting to public leaderboard noise.

### 14. Main Reflection: Failed 100+ AI Feature Experiment

At first, I asked AI to generate over 100 features and then tried to AB-test them, keeping only the ones that improved accuracy. However, the result stayed in the 0.93 range, so I discarded the entire path and rebuilt the model from scratch by understanding the problem structure.

From this, I learned that simply asking AI to generate many features does not automatically create a good model. The important part is understanding Formula 1 pit strategy and asking whether each feature explains why a car would pit on the next lap. AI is good for idea generation, but human reasoning is necessary for feature selection, leakage checking, validation design, and final interpretation.

### 15. GitHub Repository Explanation

The GitHub repository should contain code and compact reproducible artifacts, not large generated files.

| Keep in GitHub | Reason |
|---|---|
| `README.md` | Entry point and project summary. |
| `REPORT.md` | Full reflection report with images. |
| `src/make_eda_report.py` | Reproducible EDA generation. |
| modeling scripts | Preserve LightGBM/CatBoost/XGBoost/ensemble logic. |
| `requirements.txt` | Environment reproduction. |
| `reports/figures/eda/` | Images needed for GitHub report display. |
| `reports/eda_summary_tables/` | Compact EDA evidence. |
| `reports/key_results.csv` | Main score table. |
| small run summaries | Evidence for results and weights. |

| Do not keep in GitHub | Reason |
|---|---|
| `data/train.csv`, `data/test.csv` | Large raw competition files. |
| original dataset files | External data; should be downloaded locally. |
| submission CSVs | Generated artifacts. |
| OOF/test prediction arrays if large | Generated and possibly bulky. |
| model binaries | Large and reproducible. |
| `.vs/`, `__pycache__/`, cache folders | Local-only clutter. |

Recommended structure:

```text
f1Prediction/
  README.md
  REPORT.md
  requirements.txt
  .gitignore
  src/
    make_eda_report.py
    train_lgbm.py
    train_catboost.py
    ensemble.py
  reports/
    figures/eda/
    eda_summary_tables/
    key_results.csv
  data/
    README.md
```

### 16. General Lessons for Future Kaggle Competitions

1. Save OOF and test predictions for every model.
2. Build a strong simple baseline before generating many features.
3. Separate manual/domain features from AI-generated features.
4. Add features with hypotheses, not just because they are possible.
5. Check train/test/original distribution shifts.
6. Try at least one non-tree model such as RealMLP for diversity.
7. Evaluate model correlation before blending.
8. For AUC competitions, try both probability blending and rank blending.
9. Do not trust public leaderboard too much.
10. Keep a clean GitHub repo with reproducible scripts and compact artifacts.

### 17. Sources and Notes

This report is based on my saved project artifacts, EDA outputs, and reflection notes. External technical references used for model descriptions:

- LightGBM documentation: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- CatBoost documentation: https://catboost.ai/docs/en/references/training-parameters/
- XGBoost documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- Optuna documentation: https://optuna.readthedocs.io/
- PyTabKit / RealMLP repository: https://github.com/dholzmueller/pytabkit
- RealMLP paper: https://arxiv.org/abs/2407.04491



---

## 日本語レポート

### 0. 含めた項目のチェックリスト

この版では、コンペ目的、専門用語、データ概要、original data、ローカル実行理由、各列の意味、画像付きEDA、実験ログ、スコア改善・悪化要因、良かった特徴量・悪かった特徴量と数値、解釈、最終モデルでやったこと、GitHub構成、上位者との差分、RealMLP、blending/rank blending、今後のKaggleへの一般化した学びをすべて含めています。

### 1. コンペの目的

このKaggleコンペの目的は、F1のラップ単位データから、次のラップでピットインするかどうかを予測することでした。目的変数は `PitNextLap` で、0または1の二値です。1なら「次のラップでpitする」、0なら「次のラップではpitしない」という意味です。

これは **binary classification（二値分類）** です。提出では、test dataの各 `id` に対して、`PitNextLap` になる確率のような値を出します。評価指標はROC AUCなので、確率そのものの絶対値よりも、「実際にpitする行を、pitしない行より上に並べられているか」が重要です。

### 2. 登場する専門用語の解説

| 用語 | 説明 | 今回の具体例 |
|---|---|---|
| Binary classification（二値分類） | 結果が2種類の分類問題。 | `PitNextLap` が0か1。 |
| Target / objective variable（目的変数） | モデルが予測したい列。 | `PitNextLap`。 |
| Feature / explanatory variable（特徴量・説明変数） | モデルに入力する列。 | `TyreLife`, `LapNumber`, `Compound`。 |
| Raw feature（元特徴量） | CSVに最初から入っている列。 | `RaceProgress`。 |
| Engineered feature（作成特徴量） | 元の列から新しく作った特徴量。 | `Race_Compound_Stint`, rolling mean。 |
| Train data（学習データ） | 目的変数があり、学習に使うデータ。 | `train.csv`。 |
| Test data（テストデータ） | 目的変数がなく、予測を提出するデータ。 | `test.csv`。 |
| Sample submission | 提出形式のテンプレート。 | `sample_submission.csv`。 |
| Original data | Playgroundのtrain/testとは別の元データ・外部データ。 | 追加学習や分布確認に使えるが、分布差確認が必要。 |
| Synthetic data | 直接の生データではなく、生成・加工されたデータ。 | Playground系列のデータ。 |
| ROC AUC | 正例を負例より上に順位付けできるかを見る指標。 | pitする行を上に並べるほど高い。 |
| OOF prediction | Out-of-fold予測。その行を学習に使っていないモデルで予測した値。 | モデル比較やblendに使う。 |
| Fold | CVの1分割。 | 5-foldなら各行が一度validationになる。 |
| Cross-validation（交差検証） | train/validationを分けて汎化性能を見る方法。 | OOF AUCを計算する。 |
| StratifiedKFold | 目的変数の比率を保って分割するCV。 | 不均衡な二値分類で便利。 |
| Group CV | 同じグループをtrain/validationに混ぜないCV。 | Race単位で分けるとリーク対策になる可能性。 |
| Leakage（リーク） | 本来予測時に使えない未来情報が混ざること。 | 未来のpitを直接表す特徴量は危険。 |
| Feature engineering（特徴量エンジニアリング） | 予測に役立つ列を新しく作ること。 | `Race_Year`, frequency encodingなど。 |
| Frequency encoding（頻度エンコーディング） | カテゴリの出現回数・頻度を特徴量にする。 | `Driver_count`, `Race_freq`。 |
| Target encoding（ターゲットエンコーディング） | カテゴリごとの目的変数平均を特徴量化する。 | `Race_Compound`ごとのpit率。ただしCV内で安全に作る必要がある。 |
| Rolling feature（ローリング特徴量） | 直近数行の平均などを使う特徴量。 | 直近3周の`LapTime_Delta`平均。 |
| Lag feature（ラグ特徴量） | 1つ前など過去の値を使う特徴量。 | 前ラップの`LapTime_Delta`。 |
| Group statistics（グループ統計量） | groupごとの平均・標準偏差など。 | `Race_Year`ごとの`LapTime_Delta`平均。 |
| Standard deviation（標準偏差） | 値のばらつきの大きさ。 | 同じrace内でラップタイムがどれくらい散らばるか。 |
| Z-score / deviation | グループ平均との差・標準偏差何個分か。 | そのrace内で今のlapがどれくらい遅いか。 |
| Pruning | 不要そうな特徴量を落とすこと。 | importanceで150/180特徴量に削る実験。 |
| Ablation / AB test | 特徴量を足す/抜く比較実験。 | AB-selected features。 |
| Ensemble（アンサンブル） | 複数モデルを組み合わせること。 | LightGBM + CatBoost。 |
| Blending | 予測値を重み付き平均すること。 | `0.427 * LGBM + 0.573 * CatBoost`。 |
| Rank blending | 確率ではなく順位を混ぜること。 | AUC向けに有効なことがある。 |
| Stacking | OOF予測を入力にして2段目モデルを学習すること。 | Logistic Regression stackerなど。 |
| Hyperparameter（ハイパーパラメータ） | 学習前に設定する値。 | `learning_rate`, `max_depth`。 |
| Optuna | ハイパーパラメータを自動探索するツール。 | 複数の設定を試し、CV AUCが高いものを探す。 |
| Early stopping | validationが伸びなくなったら学習を止めること。 | 過学習対策。 |
| Overfitting（過学習） | 学習データやCVに合わせすぎること。 | publicでは良いがprivateで落ちるなど。 |
| Public / Private LB | Kaggleの公開/非公開leaderboard。 | 最終順位はprivateで決まる。 |
| LightGBM | 高速な勾配ブースティング木モデル。 | 強いtabular baseline。 |
| CatBoost | カテゴリ変数に強い勾配ブースティング木モデル。 | 今回の強い単体モデル。 |
| XGBoost | 伝統的で強力な勾配ブースティング木モデル。 | 今回は弱く、最終重み0。 |
| RealMLP | 表形式データ向けに改善されたMLP。 | 自分は未使用だが、上位者が使用。 |

### 3. データ概要

#### 3.1 メインのコンペファイル

| dataset           |   rows |   columns |   missing_cells |   missing_pct |   duplicate_rows |
|:------------------|-------:|----------:|----------------:|--------------:|-----------------:|
| train             | 439140 |        16 |               0 |             0 |                0 |
| test              | 188165 |        15 |               0 |             0 |                0 |
| sample_submission | 188165 |         2 |               0 |             0 |                0 |

trainは439,140行、16列で、目的変数 `PitNextLap` を含みます。testは188,165行、15列で、目的変数はありません。目的変数の陽性率は約19.90%です。

このデータは不均衡ですが、極端にまれなイベントではありません。50:50ではないというだけで、1%や0.1%のようなrare eventではありません。今回扱うモデルはLightGBMやCatBoostのような勾配ブースティングモデルなので、この程度の不均衡はAUC評価では大きな問題になりにくいです。

#### 3.2 original dataについて

KaggleのPlayground `train.csv` / `test.csv` とは別に、F1 strategy系の **original data** も参照・追加候補として扱いました。original dataは、Playgroundのsynthetic dataとは別系統のデータで、より元のF1戦略パターンを含んでいる可能性があります。

ただし、original dataはそのまま混ぜればよいわけではありません。列名、target定義、分布、欠損、外れ値、sessionの意味がcompetition dataと違う可能性があります。そのため、正しい使い方は以下です。

1. 列名と意味をcompetition dataに合わせる。
2. 予測時に使えない列を除外する。
3. train/test/originalの分布差を見る。
4. originalあり/なしでCVを比較する。
5. OOFとleaderboardの挙動が安定する場合だけ採用する。

GitHubにはraw data本体を入れず、`data/README.md`にダウンロード先と配置方法だけ書くのが良いです。

### 4. ローカルで実行したこととその理由

多くの実験はKaggle NotebookではなくローカルPCで実施しました。理由は、スクリプトを何度も回しやすく、OOF予測、feature importance、選択特徴量、calibration表、submission候補などの中間ファイルを保存しやすかったからです。

Kaggle Notebookは共有には便利ですが、実験中は容量・セッション・フォルダ整理の面でやや制約があります。したがって、今回はローカルで探索し、最後にGitHubに残すべきコード・図・小さな結果ファイルだけを整理する方針にしました。

次回の理想は、ローカルで素早く実験し、OOF/test predictionをすべて保存し、最後に再現可能なKaggle NotebookやGitHub reportにまとめる流れです。

### 5. 各列の意味とEDA上の注意

| 列                     | 意味                                                                                 | 型                         | モデリング上の注意                                           |
|:-----------------------|:-------------------------------------------------------------------------------------|:---------------------------|:-------------------------------------------------------------|
| id                     | 行ごとの一意な識別子。予測ファイルとの結合には必要だが、予測特徴量としては使わない。 | ID                         | 学習には入れない。                                           |
| Driver                 | ドライバーコードまたは匿名化ID。ドライバー・チーム・戦略の癖を含む可能性がある。     | 高カーディナリティカテゴリ | 頻度特徴量やCatBoostのカテゴリ処理と相性がよい。             |
| Compound               | HARD, MEDIUM, SOFT, INTERMEDIATE, WETなどのタイヤ種類。                              | カテゴリ                   | 戦略文脈として強い。EDAではHARDの次ラップpit率が高かった。   |
| Race                   | グランプリまたはセッション名。コース、周回数、戦略がレースごとに違う。               | カテゴリ                   | Race × Yearなどの交互作用が効きやすい。                      |
| Year                   | シーズン年。                                                                         | 順序・離散数値             | 分布差が大きいため、検証設計と合わせて慎重に扱う。           |
| PitStop                | 現在行のpit stop関連フラグ。                                                         | 二値・離散                 | 有用だが、リークに近い挙動がないか確認が必要。               |
| LapNumber              | 現在のラップ番号。                                                                   | 順序数値                   | 戦略的pit windowを表しやすく、目的変数との相関が高い。       |
| Stint                  | 現在のスティント番号。                                                               | 順序数値                   | Stint 2のpit率が高く、TyreLifeやRaceProgressと相互作用する。 |
| TyreLife               | 現在のタイヤを使った周回数。                                                         | 数値                       | このEDAで目的変数との関係が最も強い数値特徴量。              |
| Position               | 現在順位。                                                                           | 順序数値                   | 単体相関は弱いが、戦略文脈では意味を持つ。                   |
| LapTime (s)            | ラップタイム秒。                                                                     | 連続数値                   | 大きな外れ値があり、race内相対値の方が役立ちやすい。         |
| LapTime_Delta          | 基準ラップタイムとの差。                                                             | 連続数値                   | 外れ値があり、rollingや平均との差分の方が効きやすい可能性。  |
| Cumulative_Degradation | 累積の劣化・ペース低下シグナル。                                                     | 連続数値                   | 目的変数と負の相関。クリッピングや相対特徴量が重要。         |
| RaceProgress           | レースまたはセッションの進行率。                                                     | 0〜1の連続数値             | レースの局面を表す。StintやTyreLifeとの交互作用が重要。      |
| Position_Change        | 順位変化。                                                                           | 離散数値                   | 単体相関は弱いが、荒れた局面や戦略変更のサインになりうる。   |
| PitNextLap             | 目的変数。1なら次ラップでpitする。                                                   | 二値目的変数               | AUC評価用に確率として予測する。                              |

#### trainの数値・型の概要

| column                 | role    | dtype   |   missing |   unique | min        | median      | mean        | max         |
|:-----------------------|:--------|:--------|----------:|---------:|:-----------|:------------|:------------|:------------|
| id                     | id      | int64   |         0 |   439140 | 0.0000     | 219569.5000 | 219569.5000 | 439139.0000 |
| Driver                 | feature | str     |         0 |      887 |            |             |             |             |
| Compound               | feature | str     |         0 |        5 |            |             |             |             |
| Race                   | feature | str     |         0 |       26 |            |             |             |             |
| Year                   | feature | int64   |         0 |        4 | 2022.0000  | 2024.0000   | 2023.5235   | 2025.0000   |
| PitStop                | feature | int64   |         0 |        2 | 0.0000     | 0.0000      | 0.1361      | 1.0000      |
| LapNumber              | feature | int64   |         0 |       78 | 1.0000     | 19.0000     | 23.1059     | 78.0000     |
| Stint                  | feature | int64   |         0 |        8 | 1.0000     | 2.0000      | 1.7891      | 8.0000      |
| TyreLife               | feature | float64 |         0 |       78 | 1.0000     | 12.0000     | 14.1582     | 77.0000     |
| Position               | feature | int64   |         0 |       20 | 1.0000     | 10.0000     | 9.6303      | 20.0000     |
| LapTime (s)            | feature | float64 |         0 |    37719 | 67.6940    | 90.5210     | 90.9487     | 2507.6070   |
| LapTime_Delta          | feature | float64 |         0 |    57532 | -2403.8950 | -0.2950     | -3.7700     | 2423.9320   |
| Cumulative_Degradation | feature | float64 |         0 |   142701 | -274.5640  | -20.9940    | -25.7218    | 2412.0260   |
| RaceProgress           | feature | float64 |         0 |     1898 | 0.0128     | 0.2692      | 0.3377      | 1.0000      |
| Position_Change        | feature | float64 |         0 |       37 | -18.0000   | 0.0000      | 0.1015      | 18.0000     |
| PitNextLap             | target  | float64 |         0 |        2 | 0.0000     | 0.0000      | 0.1990      | 1.0000      |

### 6. EDA：何を可視化し、何が分かったか

EDAでは、`id`を除く全カラムについて可視化しました。数値カラムならヒストグラム、箱ひげ図、`PitNextLap=0/1`別の分布比較、目的変数との平均値比較、相関係数、必要に応じたbin別pit率を確認しています。カテゴリカラムなら、棒グラフ、カテゴリ別件数、カテゴリ別pit率、上位カテゴリ、必要に応じた割合確認を行っています。時系列・順序っぽいカラムでは、`LapNumber`ごとのpit率、`TyreLife`ごとのpit率、`RaceProgress`ごとのpit率、`Year`ごとの変化を重視しています。

#### 6.1 目的変数との相関

| column                 |   corr_with_target |   abs_corr_with_target |
|:-----------------------|-------------------:|-----------------------:|
| TyreLife               |             0.2735 |                 0.2735 |
| LapNumber              |             0.2671 |                 0.2671 |
| Stint                  |             0.1982 |                 0.1982 |
| RaceProgress           |             0.1855 |                 0.1855 |
| Cumulative_Degradation |            -0.1674 |                 0.1674 |
| Year                   |             0.1253 |                 0.1253 |
| PitStop                |             0.0486 |                 0.0486 |
| Position_Change        |             0.0462 |                 0.0462 |
| LapTime (s)            |            -0.0341 |                 0.0341 |
| Position               |             0.0213 |                 0.0213 |
| LapTime_Delta          |            -0.0049 |                 0.0049 |

#### 6.2 カテゴリ別件数とpit率の例

| column   | category                  |   count |   pit_rate |
|:---------|:--------------------------|--------:|-----------:|
| Compound | HARD                      |  170518 |     0.3275 |
| Compound | SOFT                      |   38744 |     0.1935 |
| Compound | INTERMEDIATE              |   17382 |     0.1523 |
| Compound | MEDIUM                    |  211141 |     0.1011 |
| Compound | WET                       |    1355 |     0.0251 |
| Race     | Chinese Grand Prix        |    7311 |     0.3886 |
| Race     | Monaco Grand Prix         |   21539 |     0.3574 |
| Race     | Spanish Grand Prix        |   20483 |     0.32   |
| Race     | Bahrain Grand Prix        |   19535 |     0.2875 |
| Race     | Belgian Grand Prix        |    9002 |     0.2804 |
| Race     | Emilia Romagna Grand Prix |   15483 |     0.2726 |
| Race     | São Paulo Grand Prix      |   11497 |     0.2537 |
| Race     | Hungarian Grand Prix      |   22481 |     0.2393 |
| Race     | Saudi Arabian Grand Prix  |   18111 |     0.2274 |
| Race     | Las Vegas Grand Prix      |   12479 |     0.2253 |
| Stint    | 2.0                       |  129536 |     0.3911 |
| Stint    | 3.0                       |   69238 |     0.2931 |
| Stint    | 4.0                       |   18903 |     0.1717 |
| Stint    | 1.0                       |  216288 |     0.0598 |
| Stint    | 5.0                       |    4281 |     0.053  |
| Stint    | 8.0                       |      50 |     0.02   |
| Stint    | 6.0                       |     728 |     0.0192 |
| Stint    | 7.0                       |     116 |     0      |
| Position | 13.0                      |   24850 |     0.2351 |
| Position | 14.0                      |   23824 |     0.2299 |
| Position | 15.0                      |   23905 |     0.2199 |
| Position | 16.0                      |   21769 |     0.2134 |
| Position | 8.0                       |   24777 |     0.2072 |
| Position | 9.0                       |   24416 |     0.2065 |
| Position | 17.0                      |   19886 |     0.2045 |
| Position | 11.0                      |   25031 |     0.2038 |
| Position | 12.0                      |   24937 |     0.2005 |
| Position | 10.0                      |   24700 |     0.1955 |


#### 目的変数の分布

目的変数 `PitNextLap` が0と1でどれくらいあるかを確認します。陽性率は約19.90%なので、不均衡ではありますが、極端にまれなイベントではありません。今回使ったLightGBMやCatBoostのような勾配ブースティングモデルでは、この程度の不均衡は大きな問題になりにくく、AUC評価でも順位付けができているかを重視できます。

![目的変数の分布](reports/figures/eda/target_distribution.png)

#### 数値カラム同士の相関と目的変数との相関

ヒートマップでは、数値カラム同士がどれくらい似た情報を持っているかを確認します。目的変数との相関では、`TyreLife`、`LapNumber`、`Stint`、`RaceProgress` が強く、pit timingをかなり直接的に表していることが分かります。一方で、`LapTime_Delta` は単純な線形相関では弱いですが、race内の平均との差分やrolling特徴量にすると意味を持つ可能性があります。

![数値変数同士の相関ヒートマップ](reports/figures/eda/correlation_heatmap.png)

![目的変数との相関](reports/figures/eda/correlation_with_target.png)

#### 各カラムごとの可視化

数値カラムでは、ヒストグラム、箱ひげ図、`PitNextLap=0/1`別の分布比較、目的変数との平均値比較、必要に応じたbin別pit率を見ます。カテゴリカラムでは、棒グラフ、カテゴリ別件数、カテゴリ別pit率、上位カテゴリ、必要に応じた円グラフ的な割合確認を行います。

![DriverのEDA](reports/figures/eda/columns/categorical_Driver.png)

![CompoundのEDA](reports/figures/eda/columns/categorical_Compound.png)

![RaceのEDA](reports/figures/eda/columns/categorical_Race.png)

![YearのEDA](reports/figures/eda/columns/numeric_Year.png)

![PitStopのEDA](reports/figures/eda/columns/numeric_PitStop.png)

![LapNumberのEDA](reports/figures/eda/columns/numeric_LapNumber.png)

![StintのEDA](reports/figures/eda/columns/numeric_Stint.png)

![TyreLifeのEDA](reports/figures/eda/columns/numeric_TyreLife.png)

![PositionのEDA](reports/figures/eda/columns/numeric_Position.png)

![LapTime (s)のEDA](reports/figures/eda/columns/numeric_LapTime_s.png)

![LapTime_DeltaのEDA](reports/figures/eda/columns/numeric_LapTime_Delta.png)

![Cumulative_DegradationのEDA](reports/figures/eda/columns/numeric_Cumulative_Degradation.png)

![RaceProgressのEDA](reports/figures/eda/columns/numeric_RaceProgress.png)

![Position_ChangeのEDA](reports/figures/eda/columns/numeric_Position_Change.png)

#### train/test分布比較

trainとtestで分布が違うと、CVでは良くてもleaderboardでは崩れる可能性があります。そのため、特徴量を追加する前に、train/test/originalの分布差を確認することが重要です。

![分布比較: Race](reports/figures/eda/dataset_comparison/compare_Race.png)

![分布比較: Compound](reports/figures/eda/dataset_comparison/compare_Compound.png)

![分布比較: Year](reports/figures/eda/dataset_comparison/compare_Year.png)

![分布比較: LapNumber](reports/figures/eda/dataset_comparison/compare_LapNumber.png)

![分布比較: TyreLife](reports/figures/eda/dataset_comparison/compare_TyreLife.png)

![分布比較: RaceProgress](reports/figures/eda/dataset_comparison/compare_RaceProgress.png)

![分布比較: LapTime (s)](reports/figures/eda/dataset_comparison/compare_LapTime_s.png)

![分布比較: LapTime Delta](reports/figures/eda/dataset_comparison/compare_LapTime_Delta.png)

![分布比較: Cumulative Degradation](reports/figures/eda/dataset_comparison/compare_Cumulative_Degradation.png)

![分布比較: Position](reports/figures/eda/dataset_comparison/compare_Position.png)

![分布比較: Position Change](reports/figures/eda/dataset_comparison/compare_Position_Change.png)

![分布比較: Stint](reports/figures/eda/dataset_comparison/compare_Stint.png)

![分布比較: PitStop](reports/figures/eda/dataset_comparison/compare_PitStop.png)

![分布比較: Driver](reports/figures/eda/dataset_comparison/compare_Driver.png)


#### 6.3 各カラムごとの解釈

| カラム | EDAで見えたこと | モデル上の意味 |
|---|---|---|
| Driver | 887種類ある高カーディナリティカテゴリ。driverごとにpit率差があるが、少数カテゴリはノイズもある。 | 頻度特徴量やCatBoostのカテゴリ処理が重要。 |
| Compound | HARDのpit率が高く、MEDIUMは低め。WETは少なく特殊。 | タイヤ戦略を直接表す。Stint/Raceとの組み合わせが重要。 |
| Race | raceごとのpit率差が大きい。Chinese/Monaco/Spanishは高く、Mexico City/Miami/USは低め。 | Race文脈は非常に重要。Race_YearやRace_Compoundが効きやすい。 |
| Year | 2023だけtarget rateが大きく異なる。 | 生成データの癖や分布差の可能性があり、慎重に扱う。 |
| PitStop | PitStop=1の方が次ラップpit率が高い。 | 有用だがリークに近い挙動がないか確認する。 |
| LapNumber | 目的変数との相関が高い。 | pit windowを表す強い特徴量。 |
| Stint | Stint 2のpit率が高い。 | 戦略フェーズを表し、TyreLifeやRaceProgressと組み合わせるべき。 |
| TyreLife | 目的変数との相関が最も高い数値特徴量。 | 古いタイヤほどpitしやすいという直感に合う。 |
| Position | 単体相関は弱いが、中団でpit率が変わる。 | 交通状況やundercut/overcut戦略と関係する可能性。 |
| LapTime (s) | 外れ値が大きい。単体相関はそこまで強くない。 | race内相対値、外れ値フラグ、rolling化が有効そう。 |
| LapTime_Delta | 単純相関は弱いが、外れ値がある。 | rawよりlag/rolling/group平均との差分が良さそう。 |
| Cumulative_Degradation | 目的変数と負の相関。外れ値も大きい。 | クリッピングや相対特徴量が重要。 |
| RaceProgress | 目的変数と正の関係。 | レース局面を表す。TyreLifeやStintとの交互作用が重要。 |
| Position_Change | 単体では弱いが、正例の平均が高い。 | 荒れた局面や戦略変更のサインになり得る。 |

### 7. 実施したこと

| 手順 | 実施内容 | 理由 | 結果 |
|---|---|---|---|
| 1 | LightGBMの初期パイプライン作成 | 表形式データで強く速いbaselineを作るため | 学習・CV・提出の流れを作成 |
| 2 | AIに100個以上の特徴量を作らせてABテスト | 短時間で多くのアイデアを試すため | 0.93台で伸びず、破棄 |
| 3 | 問題構造から作り直し | AI特徴量がF1戦略理解に結びついていなかったため | 安定したStage-2 baselineへ |
| 4 | 安全な数値・ドメイン特徴量を追加 | タイヤ寿命、ラップ番号、進行率、劣化がpit判断に関係するため | 0.944248まで改善 |
| 5 | 外れ値フラグを試行 | 極端なlap/degradationがpit兆候になる可能性 | 小幅・不安定な改善 |
| 6 | 広い特徴量追加 | race/year/contextの情報を足すため | 初期pathで0.944658 |
| 7 | 重要度ベースpruning | ノイズ特徴量を落とすため | 約0.0010〜0.0011悪化 |
| 8 | Stage-2 LightGBM baseline | コンパクトで強い特徴量セットへ | 0.955756 |
| 9 | AB-selected LightGBM | 検証で効いた特徴量だけ残す | 0.958394 |
| 10 | CatBoost追加 | カテゴリ構造を別の形で拾うため | 0.959197 |
| 11 | LightGBM/CatBoost blend | 異なる誤差を平均化するため | 0.960446 |
| 12 | XGBoost追加 | 3モデル目の多様性を試すため | 弱く、重み0.0 |
| 13 | CatBoost-2000 | 長めのCatBoostを試すため | 0.960022で最良blend未満 |

### 8. 何がスコアを改善し、何が悪化させたのか

| Experiment                            | OOF AUC   | Score impact                           | Interpretation                                                                                                 |
|:--------------------------------------|:----------|:---------------------------------------|:---------------------------------------------------------------------------------------------------------------|
| Early safe numeric feature baseline   | ~0.937000 | starting point                         | Initial broad baseline before more stable engineering.                                                         |
| Safe numeric features                 | 0.944248  | +~0.007248                             | Tyre age, race phase, lap timing and degradation were close to the real pit decision mechanism.                |
| Outlier flags                         | 0.943862  | roughly flat / slight gain in one path | Outlier flags captured abnormal laps but were sparse and unstable.                                             |
| Full feature expansion                | 0.944658  | +0.000410 vs safe numeric path         | Race/year/context features added information, but the broad set also added noise.                              |
| Stage-2 LightGBM baseline             | 0.955756  | new stronger baseline                  | A compact, cleaner feature set with better CV became much stronger than the first large AI feature attempt.    |
| LightGBM AB-selected features         | 0.958394  | +0.002638 vs Stage-2 LGBM              | Direct AB testing kept features that improved validation and removed noisy ideas.                              |
| CatBoost on selected features         | 0.959197  | +0.000803 vs AB-selected LGBM          | Different categorical/rule treatment made it the strongest single model in the main path.                      |
| LightGBM + CatBoost probability blend | 0.960446  | +0.001249 vs CatBoost single           | Two strong models made similar but not identical errors; averaging improved AUC ranking.                       |
| XGBoost lightweight model             | 0.942677  | too low to help                        | The final ensemble assigned XGBoost weight 0.0 because its predictions added more noise than useful diversity. |
| Three-model ensemble                  | 0.960446  | same as 2-model blend                  | Because XGBoost received weight 0.0, this effectively became the LightGBM/CatBoost blend.                      |
| CatBoost-2000 follow-up               | 0.960022  | -0.000424 vs best blend                | Strong standalone run, but did not beat the two-model ensemble.                                                |

大事な修正として、OOF自体が「高い/低い」のではありません。OOFは予測の作り方です。高い/低いと言うべきなのは、OOF予測から計算した **OOF AUC** です。

### 9. 良かった特徴量・悪かった特徴量・数値・理由

| 出所                   | 特徴量・手法                                          | 評価                 | 根拠・数値                                                             | 理由                                                                                                                |
|:-----------------------|:------------------------------------------------------|:---------------------|:-----------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|
| Manual / domain        | TyreLife, LapNumber, RaceProgress, Stint interactions | Good                 | TyreLife corr 0.2735; LapNumber corr 0.2671; Stage-2 baseline 0.955756 | Directly connected to pit strategy timing.                                                                          |
| Manual / domain        | Race phase and tyre-age ratios                        | Good                 | Safe numeric path improved from ~0.937 to 0.944248                     | Pit decisions depend on how old the tyres are relative to the race stage.                                           |
| Manual / domain        | Outlier flags for extreme lap time/degradation        | Mixed                | 0.943686 to 0.943862 in early experiments                              | They catch abnormal/pit-window cases, but are sparse and can overfit.                                               |
| Manual / validation    | AB-selected feature set                               | Good                 | LightGBM 0.955756 to 0.958394                                          | Testing features one by one reduced noise.                                                                          |
| Manual / ensemble      | LightGBM + CatBoost blend                             | Very good            | 0.959197 to 0.960446                                                   | Blending reduced model-specific ranking errors.                                                                     |
| AI-generated / broad   | 100+ automatically generated features                 | Bad in first attempt | Result stayed in the 0.93 range and was discarded                      | Many features were not tied to F1 strategy and added noise.                                                         |
| AI-generated / broad   | Blind large lag/rolling/encoding expansion            | Mixed                | Full expansion reached 0.944658 early, but did not become final        | Some signal existed, but validation and leakage control were not mature enough.                                     |
| AI-generated / pruning | Importance-based pruning to 150/180 features          | Bad                  | about -0.0010 to -0.0011 vs full-feature baseline                      | The feature set was not filled with obviously harmful variables; pruning may remove weak-but-complementary signals. |
| Top-notebook idea      | Frequency encoding                                    | Should have tried    | Not in my final score logs; observed in top notebooks                  | Frequency can tell the model whether a driver/race/category is common, stable, or rare/noisy.                       |
| Top-notebook idea      | Group mean/std/deviation features                     | Should have tried    | Not in my final score logs; important in top notebooks                 | They express whether a lap is fast/slow relative to the same race, year, compound, or driver context.               |
| Top-notebook idea      | RealMLP                                               | Should have tried    | Not in my final model; top notebooks used it/blended it                | Different model family could add diversity beyond tree models.                                                      |

すべての個別特徴量についてAUC差分ログが残っているわけではないため、数値が確認できるものは数値で、確認できないものは特徴量グループ単位で整理しています。ここで数値を捏造しないことも重要です。

### 10. 最終モデルで実際にスコアを上げた処理

最終的に強かった流れは以下です。

```text
Stage-2 LightGBM baseline: 0.955756
→ AB-selected LightGBM:    0.958394  (+0.002638)
→ CatBoost single model:   0.959197  (+0.000803 vs AB LGBM)
→ LGBM/Cat probability blend: 0.960446  (+0.001249 vs CatBoost)
```

スコアが上がった理由は4つです。

1. **特徴量セットをきれいにしたこと**：AI生成の大量特徴量をやめ、F1のpit戦略に結びつく特徴量へ戻した。
2. **AB特徴量選択**：それっぽい特徴量を全部採用せず、OOF AUCで効いたものだけを残した。
3. **CatBoost追加**：LightGBMとは異なるカテゴリ処理・分割で、別のパターンを拾えた。
4. **確率blend**：LightGBMとCatBoostの予測は似ているが完全には同じでなく、平均することで順位が改善した。

XGBoostはOOF AUCが0.942677と低かったため、最終的な重みは0.0になりました。弱いモデルを無理に混ぜると、ensembleの多様性ではなくノイズになることがあります。

### 11. モデルのアルゴリズム、強み・弱み、ユースケース、ハイパーパラメータ

#### 11.1 Optunaとは何か

Optunaは、ハイパーパラメータを自動で探してくれるツールです。たとえば、`learning_rate`を0.01にするか0.03にするか、`max_depth`を4にするか8にするかを、人間が手で全部試す代わりに、Optunaが何回も実験して良さそうな設定を探します。

ただし、Optunaは魔法ではありません。search space、CV設計、metric、特徴量のリーク対策、実行時間の上限は人間が決める必要があります。

#### 11.2 LightGBM

LightGBMは勾配ブースティング木モデルです。たくさんの小さな決定木を順番に作り、前の木が間違えたところを次の木が修正していきます。

中学生向けに言うと、「タイヤが古い？」「レース終盤？」「HARDタイヤ？」「Stint 2？」のような質問カードを大量に作り、それらを組み合わせてpit確率を出すモデルです。

**使いどころ**：大きめの表形式データ、数値特徴量が多い問題、速いbaseline作成、feature importance確認。

**強み**：速い、強い、Kaggle tabularで定番、特徴量が多くても扱いやすい。

**弱み**：カテゴリ変数の扱いは工夫が必要。複雑にしすぎると過学習する。

| ハイパーパラメータ                  | 意味                                               | Optuna調整可？   | コメント                                                                  |
|:------------------------------------|:---------------------------------------------------|:-----------------|:--------------------------------------------------------------------------|
| n_estimators / num_boost_round      | Number of boosting trees.                          | Yes              | Usually tuned with early stopping; too many without stopping can overfit. |
| learning_rate                       | How strongly each new tree changes the prediction. | Yes              | Lower values are safer but need more trees.                               |
| num_leaves                          | Maximum number of leaves per tree.                 | Yes              | Controls tree complexity; too high can overfit.                           |
| max_depth                           | Maximum depth of each tree.                        | Yes              | Limits complexity; can be -1 for no limit.                                |
| min_child_samples                   | Minimum data points in a leaf.                     | Yes              | Higher values regularize the model.                                       |
| subsample / bagging_fraction        | Fraction of rows used per tree.                    | Yes              | Adds randomness and reduces overfitting.                                  |
| colsample_bytree / feature_fraction | Fraction of columns used per tree.                 | Yes              | Useful when there are many features.                                      |
| lambda_l1 / lambda_l2               | L1/L2 regularization.                              | Yes              | Penalizes overly complex trees.                                           |
| objective                           | Learning task such as binary classification.       | Usually no       | Set by the problem, not a free tuning target.                             |
| metric                              | Validation metric such as AUC.                     | Usually no       | Should match competition objective.                                       |
| random_state / seed                 | Random seed.                                       | Usually no       | Fix for reproducibility; can average multiple seeds later.                |

#### 11.3 CatBoost

CatBoostも勾配ブースティング木モデルですが、カテゴリ変数の扱いが特に得意です。`Driver`、`Race`、`Compound`のような文字カテゴリが多い今回のデータとは相性が良かったです。

中学生向けに言うと、LightGBMが「質問カードをたくさん作るモデル」だとすると、CatBoostは「Monacoのような特定のrace名やHARDタイヤのようなカテゴリを、かなり上手に質問カードへ変換できるモデル」です。

**使いどころ**：カテゴリ変数が多い表形式データ、高カーディナリティカテゴリ、Kaggle tabular。

**強み**：カテゴリ処理が強い、デフォルトが比較的強い、今回のようなrace/driver/compound文脈に強い。

**弱み**：LightGBMより遅い場合がある。CV設計が甘いと大量カテゴリ特徴量で過学習して見える可能性がある。

| ハイパーパラメータ   | 意味                                    | Optuna調整可？       | コメント                                                     |
|:---------------------|:----------------------------------------|:---------------------|:-------------------------------------------------------------|
| iterations           | Number of boosting iterations/trees.    | Yes                  | Often combined with early stopping.                          |
| learning_rate        | Step size of each tree.                 | Yes                  | Lower values need more iterations.                           |
| depth                | Depth of trees.                         | Yes                  | Main complexity control; too deep can overfit.               |
| l2_leaf_reg          | L2 regularization.                      | Yes                  | Important for avoiding overfit.                              |
| bagging_temperature  | Controls Bayesian bootstrap randomness. | Yes                  | Useful for generalization.                                   |
| random_strength      | Randomness in split scoring.            | Yes                  | Can reduce overfitting.                                      |
| border_count         | Number of bins for numeric features.    | Yes                  | Affects numeric feature discretization.                      |
| cat_features         | Which columns are categorical.          | No, selected by data | Must be passed correctly; not usually optimized as a number. |
| loss_function        | Training objective such as Logloss.     | Usually no           | Determined by binary classification task.                    |
| eval_metric          | Validation metric such as AUC.          | Usually no           | Should match competition objective.                          |
| random_seed          | Seed for reproducibility.               | Usually no           | Can run multiple seeds later.                                |

#### 11.4 XGBoost

XGBoostは古くから強い定番の勾配ブースティング木モデルです。LightGBMやCatBoostとは少し違う仕組み・正則化を持っています。

中学生向けに言うと、XGBoostも「質問カード」をたくさん作るモデルですが、カードの作り方や複雑さの抑え方が違う別流派です。今回はその別流派があまり強く出ませんでした。

**使いどころ**：安定したtabular baseline、正則化を効かせたいとき、複数GBDTの比較。

**強み**：成熟している、正則化が豊富、幅広く使われる。

**弱み**：今回のように他モデルより弱い場合、blendに入れるとノイズになる。カテゴリ処理は追加工夫が必要な場合がある。

| ハイパーパラメータ             | 意味                                       | Optuna調整可？   | コメント                               |
|:-------------------------------|:-------------------------------------------|:-----------------|:---------------------------------------|
| n_estimators / num_boost_round | Number of boosting rounds.                 | Yes              | Tune with early stopping.              |
| eta / learning_rate            | Step size.                                 | Yes              | Lower is safer but slower.             |
| max_depth                      | Tree depth.                                | Yes              | Main complexity control.               |
| min_child_weight               | Minimum sum of instance weight in a child. | Yes              | Higher values regularize.              |
| subsample                      | Row sampling fraction.                     | Yes              | Reduces overfitting.                   |
| colsample_bytree               | Column sampling fraction.                  | Yes              | Useful for many features.              |
| gamma                          | Minimum loss reduction to split.           | Yes              | Higher values make splitting stricter. |
| lambda / alpha                 | L2/L1 regularization.                      | Yes              | Controls complexity.                   |
| objective                      | binary:logistic for binary probability.    | Usually no       | Set by task.                           |
| eval_metric                    | auc for validation.                        | Usually no       | Should match competition.              |
| tree_method                    | hist/gpu_hist etc.                         | Sometimes        | Often chosen by environment.           |

#### 11.5 RealMLP

RealMLPは、表形式データ向けに改善されたニューラルネットワークです。通常のMLPは表形式データで木モデルに負けやすいですが、RealMLPは前処理、正則化、学習設定、デフォルト値などを工夫して、tabular dataでも戦えるようにしたモデルです。

中学生向けに言うと、木モデルは「はい/いいえの質問をたくさんするモデル」ですが、RealMLPは「たくさんのつまみを同時に少しずつ動かして、TyreLife、RaceProgress、LapTime_Deltaなどの組み合わせ方を学ぶモデル」です。

**ユースケース**：すでにLightGBM/CatBoostが強いが、別系統モデルをblendしたいとき。中〜大規模の表形式分類・回帰。モデル多様性が欲しいKaggle終盤。

**強み**：木モデルと違う誤差を出せる、smoothな特徴量の組み合わせを拾いやすい、ensemble diversityを増やせる。

**弱み**：前処理やscalingに敏感、seedや学習時間に影響される、木モデルより解釈しづらい、環境構築がやや面倒。

| ハイパーパラメータ      | 意味                                      | Optuna調整可？                     | コメント                                                                               |
|:------------------------|:------------------------------------------|:-----------------------------------|:---------------------------------------------------------------------------------------|
| hidden layers / width   | Size and shape of the neural network.     | Yes, but defaults are often strong | Controls how much representation capacity the network has.                             |
| learning rate           | Gradient descent step size.               | Yes                                | Too high is unstable; too low is slow.                                                 |
| batch size              | Number of rows per mini-batch.            | Yes                                | Affects speed, memory, and generalization.                                             |
| epochs / max epochs     | Training passes over the data.            | Yes                                | Usually use best-epoch selection or early stopping.                                    |
| weight decay            | Regularization on neural weights.         | Yes                                | Reduces overfitting.                                                                   |
| dropout                 | Randomly disables neurons while training. | Yes                                | Can help generalization but too much hurts.                                            |
| preprocessing / scaling | Robust scaling, clipping, encoding.       | Partly no                          | Usually part of the RealMLP recipe; should not be randomly changed without validation. |
| n_refit / ensemble size | How many refits/models to average.        | Yes if resources allow             | Improves stability at extra cost.                                                      |
| random seed             | Initialization and data order.            | Usually no                         | Use for reproducibility; multiple seeds can be ensembled.                              |

### 12. 手動特徴量 vs AI生成特徴量

最初にAIへ100個以上の特徴量を作らせ、それらをABテストして精度が上がったものだけを使う実験を行いました。しかし、結果的に0.93台と低かったため、最終的にすべてを破棄し、1から作り直しました。

この経験から、AIに頼りすぎても良い結果が出るとは限らないと学びました。重要なのは、F1のpit戦略という問題構造を理解し、「なぜ次のラップでpitするのか」を説明できる特徴量を作ることです。

**手動・ドメイン理解で良かった特徴量**：TyreLife、LapNumber、RaceProgress、Stint、Compound、Race、Cumulative_Degradation、lap timing、AB-selected features。

**AI生成・AI補助で危なかった特徴量**：仮説の弱い大量lag/rolling、target encoding、digit/string、外れ値特徴量、意味の薄い組み合わせ特徴量。

AIはアイデア出しには便利ですが、採用するかどうかはOOF AUC、fold安定性、リークの有無、train/test分布差を見て判断する必要があります。

### 13. 上位者との差分分析

#### 13.1 自分のモデルの位置づけ

自分のモデルは、LightGBM、CatBoost、XGBoostを中心にした勾配ブースティング系のモデルでした。最終的にはLightGBM/CatBoost blendが強く、かなり戦えるスコアまで到達しました。

ただし上位者は、特徴量の文脈化、モデル多様性、最終blendingまでさらに踏み込んでいました。

#### 13.2 上位者がやっていて自分が足りなかったこと

| 観点 | 自分 | 上位者 | なぜ効くか |
|---|---|---|---|
| カテゴリ交互作用 | 一部のみ | `Race_Year`, `Driver_Race`, `Race_Compound_Stint`など多数 | F1戦略は単体列ではなく文脈で決まるため |
| Frequency encoding | ほぼ未使用 | driver/race/compound組み合わせの頻度を使用 | よく出るdriverは戦略が安定し、rare patternはノイズと判断できる |
| Group stats | 限定的 | race/year/compound/stint/driverごとの平均・標準偏差 | 「そのrace内で普通より速い/遅い」を表せる |
| 標準偏差特徴量 | 不十分 | group stdを使用 | そのlapが周囲よりどれだけ異常か分かる |
| Lag/rolling | finalでは弱い | 前lap・直近3lap平均など | pitは劣化の流れで起きるため |
| Target encoding | 中心ではない | fold-safeに使用 | カテゴリごとのpitしやすさを表す |
| 特徴量数 | コンパクト | とあるシルバーメダル獲得した上位CatBoost notebookでは約300特徴量 | CatBoost + early stopping +意味ある特徴量なら成立する |
| RealMLP | 未使用 | 単体・blendで使用 | 木モデルと違う予測を出せる |
| Blending | 確率weighted blend中心 | OOF相関、rank blend、safe blend、stacking | AUCでは順位調整が重要 |

#### 13.3 頻度特徴量がなぜ良いのか

Frequency encodingは、カテゴリの出現回数を特徴量にする方法です。例えば`Driver_count`は、そのdriverがデータに何回出てきたかを表します。

これは、「そのカテゴリの情報をどれくらい信頼していいか」をモデルに伝えます。よく大会に出ているdriverは、データ上のサンプル数が多く、戦略パターンが固まっている可能性があります。一方、出現回数が少ないdriver/race組み合わせは、pit率が高く見えても偶然かもしれません。

#### 13.4 標準偏差・偏差値的特徴量がなぜ良いのか

`LapTime_Delta = 3` だけでは、それが本当に遅いのか分かりません。raceによってlap timeのばらつきが違うからです。

```text
Race A 平均delta = 0.5, 標準偏差 = 0.8, 今のdelta = 3.0 → かなり異常
Race B 平均delta = 1.5, 標準偏差 = 3.0, 今のdelta = 3.0 → そこまで異常ではない
```

標準偏差や平均との差分を作ることで、「そのrace内のそのlapが周囲よりどれだけ速い/遅いか」が分かります。これはrawなlap timeよりも戦略判断に近い特徴量です。

#### 13.5 なぜ300特徴量でも過学習しすぎなかったのか

とあるシルバーメダル獲得した上位CatBoost notebookでは、最終的に約300個の特徴量を使っていました。これは「特徴量は多いほど良い」という意味ではありません。

成立した理由は、主に以下です。

1. CatBoostがカテゴリ特徴量に強い。
2. 特徴量がrace/driver/compound/stintなどの文脈に沿っていた。
3. Early stoppingで学習しすぎを防いでいた。
4. 木モデルは不要特徴量を完全に均等には使わず、有用な分割を優先する。
5. foldごとのAUCが安定していた可能性がある。

ただし、CV設計が甘いと300特徴量は過学習の温床になります。重要なのは特徴量数ではなく、検証設計と特徴量の意味です。

#### 13.6 RealMLPを試さなかった反省

RealMLPを試さなかったことは大きな反省点です。自分のモデルはLightGBM、CatBoost、XGBoostという木モデル中心でした。これらは強いですが、予測の相関が高くなりやすいです。

RealMLPはニューラルネット系なので、木モデルとは違う誤差を出す可能性があります。ensembleでは、単体性能だけでなく「他のモデルと違う間違い方をするか」が重要です。

#### 13.7 OOF保存、相関、重み探索

上位者は、各モデルのOOF予測とtest予測を保存し、モデル間の相関や重みを見ながらensembleしていました。

例えば以下のように保存します。

```text
model_lgbm_oof.csv
model_catboost_oof.csv
model_realmlp_oof.csv
model_xgb_oof.csv
```

そして相関を見ます。

```text
corr(LightGBM, CatBoost) = 0.97
corr(LightGBM, RealMLP)  = 0.92
corr(CatBoost, RealMLP)  = 0.91
```

この数値は説明用の例ですが、考え方が重要です。相関が高すぎるモデルを混ぜても新しい情報は少ないです。一方で、AUCがそこそこ高く、相関が低めのモデルはblendで効く可能性があります。

重み探索は以下のように行います。

```text
final_oof = 0.45 * lgbm_oof + 0.40 * cat_oof + 0.15 * realmlp_oof
```

OOF AUCが最も高い重みを探し、同じ重みをtest predictionに適用します。

#### 13.8 Blending / rank blending の学び

通常のblendingは確率を平均します。

```text
final = 0.7 * model_A_probability + 0.3 * model_B_probability
```

rank blendingは、確率そのものではなく順位を平均します。

```text
rank_A = rank(model_A_probability)
rank_B = rank(model_B_probability)
final_rank_score = 0.7 * rank_A + 0.3 * rank_B
```

AUCは順位が大事なので、rank blendingが効く場合があります。確率のスケールが少し変でも、順位が良いsubmissionなら活用できます。

上位者は、Kaggle内の過去の似たtabular / AUC系コンペやpublic notebookから、どのようなblendが安全かを学び、それを今回のOOFで検証していました。大事なのは、過去手法を丸コピーすることではなく、OOFで自分のデータに合うか確認することです。

### 14. 最重要の反省：AIに100個以上の特徴量を作らせた実験

最初にAIへ100個以上の特徴量を作らせ、それらをABテストして精度が上がったものだけを残す実験を行いました。しかし、結果は0.93台にとどまり、最終的にその流れをすべて破棄して、問題構造を理解するところから作り直しました。

この経験から、AIに大量の特徴量を作らせるだけでは良いモデルにはならないと学びました。重要なのは、F1のピット戦略という問題の構造を理解し、どの特徴量が「次のラップでピットする理由」とつながるのかを考えることでした。

AIはアイデア出しには便利ですが、最終的には人間が問題構造、リーク、CV設計、実験結果を見て判断する必要があります。

### 15. GitHubに入れるファイルの考え方

GitHubには、再現性と読みやすさに必要なものだけを入れます。

| GitHubに入れるもの | 理由 |
|---|---|
| `README.md` | プロジェクト概要。 |
| `REPORT.md` | 画像付き最終レポート。 |
| `src/make_eda_report.py` | EDAを再生成するコード。 |
| モデリングスクリプト | LightGBM/CatBoost/XGBoost/ensembleの処理を残す。 |
| `requirements.txt` | 環境再現。 |
| `reports/figures/eda/` | GitHub上で画像を表示するため。 |
| `reports/eda_summary_tables/` | EDA根拠の小さな表。 |
| `reports/key_results.csv` | 主要スコア表。 |
| 小さなrun summary | 結果や重みの根拠。 |

| GitHubに入れないもの | 理由 |
|---|---|
| `data/train.csv`, `data/test.csv` | 大きなraw competition file。 |
| original dataset本体 | 外部データなのでlocalで配置。 |
| submission CSV | 生成物。 |
| 大きなOOF/test prediction配列 | 生成可能で重い。 |
| model binary | 大きく再生成可能。 |
| `.vs/`, `__pycache__/` | ローカル専用。 |

推奨構成は以下です。

```text
f1Prediction/
  README.md
  REPORT.md
  requirements.txt
  .gitignore
  src/
    make_eda_report.py
    train_lgbm.py
    train_catboost.py
    ensemble.py
  reports/
    figures/eda/
    eda_summary_tables/
    key_results.csv
  data/
    README.md
```

### 16. 今後のKaggleで使える一般化した学び

1. 各モデルのOOF predictionとtest predictionを必ず保存する。
2. 大量特徴量の前に、強いシンプルbaselineを作る。
3. 手動特徴量とAI生成特徴量を分けて管理する。
4. 特徴量は「なぜ目的変数に効くか」という仮説つきで作る。
5. train/test/originalの分布差を見る。
6. LightGBM/CatBoostだけでなくRealMLPのような非木モデルも試す。
7. blend前にモデル間相関を見る。
8. AUCコンペではprobability blendingだけでなくrank blendingも試す。
9. public leaderboardに寄せすぎない。
10. GitHubには再現可能なコード・図・小さな結果だけを入れる。

### 17. 参考・メモ

このレポートは、自分の保存済みproject artifact、EDA出力、上位notebookの振り返りメモをもとに作成しています。モデル説明で参照した外部資料：

- LightGBM documentation: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- CatBoost documentation: https://catboost.ai/docs/en/references/training-parameters/
- XGBoost documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- Optuna documentation: https://optuna.readthedocs.io/
- PyTabKit / RealMLP repository: https://github.com/dholzmueller/pytabkit
- RealMLP paper: https://arxiv.org/abs/2407.04491
