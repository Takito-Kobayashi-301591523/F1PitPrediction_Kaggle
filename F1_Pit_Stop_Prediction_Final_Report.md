# F1 Pit Stop Prediction Final Reflection Report

This Markdown file is designed for GitHub. It contains the full English report first, then the Japanese report, followed by the auto-generated EDA appendix with image links.

To display all images correctly on GitHub, keep this file in the repository root and keep the `reports/` folder next to it.

## English Report

### 1. Competition Objective

The goal of this Kaggle competition was to predict `PitNextLap`, a binary target that indicates whether an F1 car will pit on the next lap. Each row represents a lap-level situation: driver, race, tyre compound, stint, lap number, tyre life, race progress, position, lap time, lap-time delta, and degradation-related variables.

The evaluation metric was ROC AUC. This matters because AUC mainly evaluates ranking: among two rows, one that actually pits next lap and one that does not, the model should assign a higher score to the pit row. Therefore, the project was not only about estimating perfectly calibrated probabilities; it was about ranking pit-risk situations correctly.

My final private result was **0.95152**, which placed around **782nd out of roughly 3,000 participants**. The final solution was competitive, but the top notebooks showed several clear improvement opportunities: richer contextual features, RealMLP, more systematic OOF management, model-correlation analysis, and more advanced blending.

### 2. Data Overview

| Dataset | Role | Notes |
| --- | --- | --- |
| `train.csv` | Training data | Contains all explanatory columns and the target `PitNextLap`. |
| `test.csv` | Test data | Contains the same explanatory columns but not the target. |
| `sample_submission.csv` | Submission template | Contains `id` and the required prediction column. |
| Original F1 strategy dataset | External/original data source | Used as a reference or optional additional data source. It should not be committed to GitHub; instead, the repository should explain where users can place it locally. |

The training target was imbalanced but not extremely rare. In the generated EDA, `PitNextLap = 1` appeared in about **19.90%** of the training rows. This is not a 50:50 classification problem, but it is also not an ultra-rare event. Models such as LightGBM and CatBoost can usually handle this level of imbalance reasonably well, especially when the evaluation metric is AUC rather than raw accuracy.

The original dataset was important because it represented the source-style F1 pit strategy data behind the Playground competition. However, the uploaded package did not include the raw original CSV itself, so this report describes it as an external/reference dataset rather than giving exact row and column counts. The EDA script supports an `original` folder so that, when the original files are placed locally, train/test/original distribution comparison charts can be generated automatically.

### 3. Glossary

| Term | Explanation |
| --- | --- |
| Binary classification | A prediction problem with two possible outcomes. Here, `PitNextLap` is 0 or 1. |
| ROC AUC | A ranking metric. It measures how often positive rows are ranked above negative rows. |
| OOF prediction | Out-of-fold prediction. Each training row is predicted by a model that did not train on that row. This makes it useful for honest validation, blending, and stacking. |
| Feature engineering | Creating new useful columns from existing data. Example: `TyreLife / LapNumber` or `Race_Compound_Stint`. |
| Tag feature | A feature that labels a situation with a meaningful group or bucket. Example: `race_phase = early/mid/late` or `tyre_life_bin = young/medium/old`. |
| Rolling feature | A feature computed over recent rows in a sequence. Example: the average `LapTime_Delta` over the previous 3 laps for the same driver/race/stint. |
| Target encoding | Replacing a category with its historical target rate, usually computed carefully inside cross-validation to avoid leakage. Example: average `PitNextLap` rate for each `Race_Compound` group. |
| Frequency encoding | Replacing or adding a category's count/frequency. Example: how often `Driver_Race` appears. It tells the model whether a category is common, stable, or rare/noisy. |
| Group statistics | Mean, standard deviation, min, max, or difference from group average. Example: `LapTime_Delta_std_by_Race_Year`. |
| Blending | Combining predictions from multiple models or submissions, often by weighted average or rank average. |
| Stacking | Training a second-level model using OOF predictions from multiple base models as features. |
| Leakage | Using information that would not be available at prediction time. Strong features must be checked carefully. |

### 4. EDA Report and Visual Findings

The full auto-generated EDA is included in **Appendix A** and keeps all generated plots from `reports/figures/eda/`. The key findings were:

1. `TyreLife`, `LapNumber`, `Stint`, and `RaceProgress` had the strongest simple relationships with the target.
2. `Compound` and `Race` were highly informative because pit strategy changes by circuit and tyre type.
3. `Year` had a suspiciously different distribution, so it needed careful validation.
4. `LapTime (s)`, `LapTime_Delta`, and `Cumulative_Degradation` had large outliers, which suggested the need for robust clipping, outlier flags, or group-relative statistics.
5. Train/test/original comparison plots were important because a feature can look strong locally but fail if the test distribution is different.

**Target distribution**

![Target distribution](reports/figures/eda/target_distribution.png)

**Numeric correlation heatmap**

![Numeric correlation heatmap](reports/figures/eda/correlation_heatmap.png)

**Correlation with target**

![Correlation with target](reports/figures/eda/correlation_with_target.png)

**Compound: frequency and target rate**

![Compound: frequency and target rate](reports/figures/eda/columns/categorical_Compound.png)

**Race: frequency and target rate**

![Race: frequency and target rate](reports/figures/eda/columns/categorical_Race.png)

**Driver: frequency and target rate**

![Driver: frequency and target rate](reports/figures/eda/columns/categorical_Driver.png)

**Year: distribution and target relationship**

![Year: distribution and target relationship](reports/figures/eda/columns/numeric_Year.png)

**LapNumber: distribution and target relationship**

![LapNumber: distribution and target relationship](reports/figures/eda/columns/numeric_LapNumber.png)

**Stint: distribution and target relationship**

![Stint: distribution and target relationship](reports/figures/eda/columns/numeric_Stint.png)

**TyreLife: distribution and target relationship**

![TyreLife: distribution and target relationship](reports/figures/eda/columns/numeric_TyreLife.png)

**RaceProgress: distribution and target relationship**

![RaceProgress: distribution and target relationship](reports/figures/eda/columns/numeric_RaceProgress.png)

**LapTime Delta: distribution and target relationship**

![LapTime Delta: distribution and target relationship](reports/figures/eda/columns/numeric_LapTime_Delta.png)

**Cumulative Degradation: distribution and target relationship**

![Cumulative Degradation: distribution and target relationship](reports/figures/eda/columns/numeric_Cumulative_Degradation.png)

**Train/test/original comparison: Race**

![Train/test/original comparison: Race](reports/figures/eda/dataset_comparison/compare_Race.png)

**Train/test/original comparison: Compound**

![Train/test/original comparison: Compound](reports/figures/eda/dataset_comparison/compare_Compound.png)

### 5. What I Did in My Final Modeling Path

| Step | What I did | Why it helped or did not help |
| --- | --- | --- |
| LightGBM baseline | Built a reliable tabular baseline. | LightGBM is fast and strong on tabular data, so it gave a solid starting point. |
| Basic domain features | Added tyre-life ratios, race-progress interactions, lap/time differences, and degradation-related features. | These features directly describe why a driver might pit: old tyres, race phase, pace loss, and position context. |
| Outlier flags | Tested flags for abnormal lap time, lap-time delta, and degradation. | Some flags had high target lift, but many were sparse and unstable, so the gain was limited. |
| AB feature selection | Tested candidate features one by one and kept features that improved validation. | This prevented blindly trusting every AI-generated feature. The compact selected set improved LightGBM. |
| CatBoost | Trained CatBoost on the selected features. | CatBoost handled categorical/rule-like structure differently from LightGBM and became a strong single model. |
| LightGBM + CatBoost blending | Searched weights using OOF predictions. | The best OOF blend used both models and reduced model-specific errors. |
| XGBoost | Tested a lightweight XGBoost as a third model. | It was weaker in OOF, so the weight search assigned it zero weight. |
| CatBoost-2000 follow-up | Tried stronger CatBoost training. | It was strong but did not beat the best two-model blend. |

A key correction: it is not accurate to say “OOF was low.” OOF is a validation prediction method. The thing that was low was **OOF AUC**.

### 6. Main Results

| Experiment | OOF AUC | Interpretation |
| --- | ---: | --- |
| Stage-2 LightGBM baseline | 0.955756 | Strong baseline from core features. |
| LightGBM AB-selected | 0.958394 | Feature selection improved LightGBM. |
| CatBoost on selected features | 0.959197 | Strongest single model in the main path. |
| LightGBM + CatBoost probability blend | 0.960446 | Best local OOF result. |
| XGBoost lightweight model | 0.942677 | Too weak to help the final blend. |
| Three-model ensemble | 0.960446 | XGBoost received weight 0.0, so this matched the two-model blend. |
| CatBoost-2000 follow-up | 0.960022 | Strong, but still slightly below the best two-model blend. |

### 7. Reflection on the Failed “100+ AI Features” Attempt

Before building the final modeling path, I asked AI to create more than 100 features and then ran AB tests to keep only features that improved score. The result stayed around the 0.93 range, so I discarded that entire path and rebuilt the solution from scratch.

The lesson is not “AI is useless.” The lesson is that feature engineering must start from the problem mechanism. In this competition, a useful feature should connect to a real pit decision factor: tyre age, stint, lap number, race progress, compound, circuit, position, pace loss, or recent degradation. AI can generate ideas quickly, but if those ideas are not filtered through domain logic and stable validation, they can add noise.

General Kaggle lesson: **feature quantity is not the same as feature quality**. A good feature explains the target mechanism and improves OOF AUC consistently.

### 8. Lessons from Top Notebooks

The uploaded reflection file summarized several top-notebook differences. The biggest gaps were not just “better models,” but a more complete competition pipeline: contextual features, group statistics, frequency encoding, RealMLP, saved OOF predictions, correlation-aware ensembling, and final-stage blending.

#### 8.1 Contextual categorical interactions

Top notebooks created features such as:

- `Race_Year`
- `Driver_Race`
- `Driver_Compound`
- `Race_Compound`
- `Compound_Stint`
- `Race_Compound_Stint`
- `RacePhase_TyreLifeBin`
- `Compound_TyreLifeBin`

These features matter because the same value can mean different things in different contexts. For example, `HARD` tyres at Monaco and `HARD` tyres at Bahrain do not imply the same pit strategy. A `Stint=2` row early in the race is not the same as a `Stint=2` row near the end. Top notebooks made those contexts explicit.

#### 8.2 Group statistics and standard deviation features

Top CatBoost notebooks created features like:

- `Position_Change_std_by_Race_Year`
- `LapTime_Delta_diff_mean_by_Race_Year`
- `RaceProgress_diff_mean_by_Compound_Stint`
- `TyreLife_diff_mean_by_Race_Compound_Stint`
- `LapTime_Delta_std_by_Driver_Race`
- `RaceProgress_mean_by_Driver_Race`

The important idea is to compare a row against its context. A raw `LapTime_Delta = 3` might look slow, but if every lap in that race has large deltas, it may not be unusual. A standard deviation feature tells the model how volatile that group is. A difference-from-mean feature tells the model whether the current lap is faster or slower than nearby/contextual expectations.

Concrete example:

| Row | Race | LapTime_Delta | Race mean | Race std | Z-like interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| A | Monaco | 3.0 | 2.5 | 0.4 | Much slower than usual for that race. |
| B | Bahrain | 3.0 | 2.9 | 2.0 | Not very unusual. |

The raw value is the same, but the meaning is different. This is why standard deviation and group-relative features can help.

#### 8.3 Frequency encoding

Frequency encoding was one of the most impressive ideas because it was easy to miss. It adds information such as:

- `Driver_count`
- `Driver_freq`
- `Race_freq`
- `Driver_Race_freq`
- `Race_Year_freq`
- `Compound_TyreLifeBin_freq`

Frequency tells the model how common or rare a pattern is. For example, if a driver appears frequently across many races, that driver has more historical evidence in the dataset. In real F1 terms, frequent appearances may also correspond to established drivers with more stable race strategy patterns. In synthetic or anonymized data, the safer interpretation is: frequent categories have more reliable statistical estimates, while rare categories are noisier and more likely to overfit.

#### 8.4 Lag and rolling features

Strong notebooks also used time-sequence features within groups such as driver × race × stint:

- `Delta_lag1`
- `Delta_roll3_mean`
- `Deg_diff`
- `TyreLife_growth`
- `LapTime_lag1`
- `LapTime_diff`

A pit stop is rarely random. It often follows a pattern: tyres age, lap time worsens, degradation increases, and the driver approaches a strategy window. Lag and rolling features capture that trend instead of looking only at the current row.

#### 8.5 Digit/string signature features

Some top CatBoost notebooks used Kaggle-style features such as rounded numeric strings or digit signatures:

- `TyreAgeRatio_str`
- `EstimatedTotalLaps_str`
- `Cumulative_Degradation_sig_4`
- `LapTime (s)_sig_3`
- `Position_Change_int_digit_1`

These are not always natural in real business analytics, but Playground competitions sometimes contain synthetic-data artifacts. Rounded values, digits, or string representations can accidentally capture patterns left by the data generation process. These features should be used carefully because they can overfit.

#### 8.6 “A silver-medal-winning CatBoost notebook used about 300 features”

In one **silver-medal-winning CatBoost notebook**, the final model used about **300 features**. This sounded risky at first, but it worked because the features were not random. They included systematic interactions, group statistics, frequency features, lag/rolling signals, and CatBoost-friendly categorical features.

The lesson is not “always use 300 features.” The lesson is: many features can work when the model is robust, early stopping is used, fold scores are stable, and the features describe meaningful contexts.

### 9. RealMLP: Algorithm, Use Cases, Strengths, Weaknesses, and Hyperparameters

RealMLP is a tabular neural-network model available through PyTabKit. It is an improved multilayer perceptron (MLP) designed for tabular data, with tuned defaults and practical training tricks. Unlike LightGBM, XGBoost, and CatBoost, which are tree-based models, RealMLP learns through layers of neurons.

#### 9.1 How the algorithm works, in beginner-friendly terms

A normal MLP works like a set of connected calculators:

1. The input row enters the model: `TyreLife`, `RaceProgress`, `Compound`, `Race`, etc.
2. Each layer combines those inputs using weights.
3. Nonlinear activation functions let the model learn curved relationships.
4. The final layer outputs a probability such as `0.73` for `PitNextLap = 1`.

For example, a tree model might learn a rule like:

```text
if TyreLife > 20 and RaceProgress > 0.55 and Compound == HARD:
    pit probability is high
```

An MLP instead learns a smoother combination:

```text
pit risk gradually increases as tyre age, degradation, and race progress rise together
```

RealMLP improves the practical MLP setup for tabular data using tuned defaults and training procedures, so it is more competitive than a plain, manually configured neural network.

#### 9.2 Why RealMLP could have helped this competition

My final models were mostly tree-based. LightGBM, CatBoost, and XGBoost are different, but they still belong to the same broad family: gradient-boosted decision trees. Their predictions can become highly correlated.

RealMLP could help because it may make different mistakes. In an ensemble, a model with slightly lower solo score can still improve the final result if its errors are different from the tree models.

Concrete example:

| Row | True target | LightGBM | CatBoost | RealMLP |
| --- | ---: | ---: | ---: | ---: |
| Old tyres, late race | 1 | 0.86 | 0.83 | 0.80 |
| Rare driver/race combo | 1 | 0.42 | 0.45 | 0.68 |
| Weird lap-time outlier | 0 | 0.75 | 0.70 | 0.40 |

If RealMLP is better on rare combinations or smoother numerical patterns, blending it with tree models can improve ranking.

#### 9.3 Good use cases

RealMLP is worth trying when:

- the dataset is tabular and medium-to-large;
- there are many continuous numerical features;
- tree models are already strong but highly correlated;
- the competition is close and ensemble diversity matters;
- GPU is available;
- OOF predictions are saved for blending/stacking;
- the target relationship may be smooth rather than only rule-based.

#### 9.4 Strengths

| Strength | Explanation |
| --- | --- |
| Different model family | Adds diversity against LightGBM/CatBoost/XGBoost. |
| Good for smooth interactions | Can learn gradual relationships among tyre age, degradation, race progress, and pace. |
| Useful for ensembling | Even if solo performance is similar, different errors can improve the blend. |
| Tuned defaults | PyTabKit provides tuned RealMLP variants, reducing manual tuning burden. |
| GPU-friendly | Neural networks can benefit from GPU acceleration. |

#### 9.5 Weaknesses

| Weakness | Explanation |
| --- | --- |
| More sensitive to preprocessing | Scaling, missing values, categorical handling, and validation setup matter. |
| Slower than simple LightGBM runs | Neural-network training can take longer. |
| Less interpretable | Feature importance is less straightforward than tree-based models. |
| Can overfit | Especially with many features, weak validation, or too many epochs. |
| More environment friction | PyTorch/PyTabKit/GPU setup may be harder than basic sklearn-style models. |

#### 9.6 Hyperparameters to know

A practical RealMLP tuning checklist:

| Hyperparameter / setting | Meaning | Practical advice |
| --- | --- | --- |
| `learning_rate` / `lr` | Step size during training. | Often one of the most important settings. Too high is unstable; too low is slow. |
| hidden width | Number of units in hidden layers. | Larger width can learn more patterns but can overfit and take longer. |
| number of layers | Depth of the neural network. | More layers can model complex patterns but may be harder to train. |
| batch size | Rows processed at once. | Larger batches can be faster on GPU; smaller batches may generalize differently. |
| epochs / max epochs | Maximum training passes. | Use early stopping instead of blindly training to the maximum. |
| early stopping / best epoch | Stops when validation score no longer improves. | Essential for tabular neural nets. |
| weight decay | Regularization on weights. | Helps reduce overfitting. |
| dropout | Randomly disables parts of the network during training. | Can help but may hurt if overused. |
| validation fraction / fold split | How validation is created. | Must match the competition structure as much as possible. |
| random seed | Controls randomness. | Train multiple seeds if the model is unstable. |
| ensemble / `n_refit` style settings | Refit or average multiple neural networks. | Useful if single RealMLP predictions are noisy. |
| categorical/numerical embedding settings | How categorical/numerical inputs are represented. | Important when many high-cardinality categories exist. |

Best practical starting point: use PyTabKit’s tuned-default RealMLP classifier first, save OOF/test predictions, then decide whether HPO or multi-seed ensembling is worth the time.

### 10. OOF Prediction Saving, Model Correlation, and Weight Search

Top solutions treated OOF predictions as reusable assets. This is one of the biggest workflow differences.

For each model, save:

```text
oof_lgbm.csv
oof_catboost.csv
oof_realmlp.csv
test_lgbm.csv
test_catboost.csv
test_realmlp.csv
fold_metrics.csv
```

Then compare model correlation:

| Pair | OOF correlation | Interpretation |
| --- | ---: | --- |
| LightGBM vs CatBoost | 0.98 | Very similar; blend gain may be limited but still possible. |
| LightGBM vs RealMLP | 0.90 | More different; potentially useful ensemble diversity. |
| CatBoost vs weak XGBoost | 0.85 | Different, but if XGBoost AUC is low, it may still add noise. |

The key is that diversity alone is not enough. A model must be both reasonably strong and different.

Concrete weight search example:

```python
best_auc = -1
best_w = None
for w in np.linspace(0, 1, 101):
    blend_oof = w * oof_lgbm + (1 - w) * oof_cat
    auc = roc_auc_score(y, blend_oof)
    if auc > best_auc:
        best_auc = auc
        best_w = w

blend_test = best_w * test_lgbm + (1 - best_w) * test_cat
```

In my project, the best local OOF result used a LightGBM/CatBoost probability blend with weights around **LightGBM 0.427 / CatBoost 0.573**.

### 11. Blending: Concrete Methods

#### 11.1 Probability blending

This is the simplest method:

```text
final = 0.60 * CatBoost + 0.40 * LightGBM
```

It works well when both models produce reasonably calibrated probabilities.

#### 11.2 Rank blending

AUC cares about order, not exact probability. Rank blending first converts predictions into ranks:

```text
Model A probabilities: 0.10, 0.90, 0.40
Model A ranks:         1,    3,    2
```

Then it averages ranks instead of probabilities. This can help when two models have useful ordering but different probability scales.

#### 11.3 Rank-remap blending

Rank-remap blending uses one strong submission as an anchor distribution and another model only for rank information. The idea is:

1. Take the rank order from a candidate model.
2. Take the probability distribution shape from a trusted anchor submission.
3. Remap the anchor values according to the candidate ranks.
4. Blend the remapped predictions with the anchor.

This can be useful in AUC competitions because it changes ordering while keeping a stable prediction distribution.

#### 11.4 Stacking with OOF predictions

Stacking trains a small second-level model using OOF predictions as features:

| Row | LGBM OOF | CatBoost OOF | RealMLP OOF | Target |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.20 | 0.25 | 0.18 | 0 |
| 2 | 0.80 | 0.75 | 0.85 | 1 |
| 3 | 0.45 | 0.50 | 0.30 | 0 |

A logistic regression stacker can learn how much to trust each model. This must use OOF predictions, not in-sample predictions, or it will overfit.

#### 11.5 Using Kaggle notebooks and similar competitions

A practical Kaggle workflow is:

1. Search the current competition for high-scoring public notebooks.
2. Search similar past Playground or tabular competitions to learn common ensemble patterns.
3. Reuse ideas such as OOF saving, rank blending, and weight search code structure.
4. If using public notebook outputs from the current competition, check competition rules and make sure the approach is allowed.
5. Do not blindly copy; test whether the candidate prediction adds OOF or leaderboard value.

Past competitions are especially useful for methodology. They usually cannot provide a submission for the current test set, but they can teach blending templates, CV structure, and feature patterns.

### 12. What I Would Do Next Time

1. Build LightGBM and CatBoost baselines quickly.
2. Save OOF and test predictions from every model from day one.
3. Create contextual categorical interactions early: `Race_Year`, `Race_Compound_Stint`, `Driver_Race`.
4. Add frequency encoding for high-cardinality categories.
5. Add group mean/std/difference features to compare each row with its race/driver/stint context.
6. Add lag and rolling features within driver × race × stint.
7. Try RealMLP early, not at the very end.
8. Check model correlations before blending.
9. Try probability blending, rank blending, rank-remap blending, and stacking.
10. Keep a conservative safe blend for final submission rather than trusting only the best OOF blend.

### 13. Final Reflection

My final solution was solid, but it was still too focused on gradient boosting. The top solutions treated the competition as a full pipeline: domain-aware feature engineering, model diversity, OOF asset management, correlation-aware ensembling, and final-stage submission blending.

The biggest general lesson is:

> Strong Kaggle solutions are rarely just one good model. They are usually a system: features, validation, diverse models, saved predictions, careful blending, and conservative final selection.

---

## Japanese Report / 日本語レポート

### 1. コンペの目的

このコンペの目的は、F1の各ラップ情報をもとに、次のラップでピットインするかどうかを表す二値目的変数 `PitNextLap` を予測することでした。各行には、ドライバー、レース、タイヤコンパウンド、スティント、ラップ番号、タイヤ寿命、レース進行率、順位、ラップタイム、ラップタイム差、劣化に関する情報が入っています。

評価指標は ROC AUC です。AUC は、確率の絶対値そのものよりも「ピットする行を、ピットしない行より上に並べられているか」を見る指標です。つまり、このコンペでは、ただ確率をそれっぽく出すだけでなく、ピットしそうな状況を正しく上位に並べることが重要でした。

最終的な自分の private score は **0.95152** で、約3,000人中 **782位前後** でした。十分に戦えた結果ではありますが、上位 notebook を見ると、まだ改善できた点がかなり明確に見えました。特に差が大きかったのは、文脈を表す特徴量、RealMLP、OOF予測の保存、モデル間相関を見たアンサンブル、そして最終段階のブレンディングです。

### 2. データ概要

| データ | 役割 | メモ |
| --- | --- | --- |
| `train.csv` | 学習データ | 説明変数と目的変数 `PitNextLap` を含む。 |
| `test.csv` | テストデータ | 説明変数のみ。目的変数は含まれない。 |
| `sample_submission.csv` | 提出テンプレート | `id` と提出用の `PitNextLap` 列を持つ。 |
| Original F1 strategy dataset | 外部/元データ | Playground の synthetic data の元になった系統のデータとして、分布比較や追加学習候補に使える。GitHubには入れず、ローカル配置方法をREADMEに書くのがよい。 |

学習データの目的変数は不均衡ですが、極端にまれなイベントではありません。自動生成EDAでは、`PitNextLap = 1` は学習データの約 **19.90%** でした。50:50ではないものの、1%未満のような超レアイベントではありません。今回使った LightGBM や CatBoost のような勾配ブースティングモデルは、この程度の不均衡には比較的対応しやすく、AUCで評価する限り大きな問題にはなりにくいです。

なお、今回アップロードされたパッケージには original CSV 本体は含まれていません。そのため、このレポートでは original data を「外部/参考データ」として説明し、正確な行数・列数は書いていません。ただし、EDAスクリプトは `original` フォルダを指定できるようにしているため、ローカルに original data を置けば train/test/original の分布比較グラフを自動で作れます。

### 3. 用語解説

| 用語 | 説明 |
| --- | --- |
| 二値分類 / Binary classification | 結果が2種類の予測問題です。今回は `PitNextLap` が 0 か 1 です。 |
| ROC AUC | 正例を負例より上に並べられているかを見る順位指標です。 |
| OOF予測 / Out-of-fold prediction | その行を学習に使っていないモデルで出した予測です。正直な検証、ブレンド、スタッキングに使えます。 |
| 特徴量エンジニアリング / Feature engineering | 元の列から予測に役立つ新しい列を作ることです。例：`TyreLife / LapNumber` や `Race_Compound_Stint`。 |
| タグ特徴量 / Tag feature | 状況に意味のあるラベルをつける特徴量です。例：`race_phase = early/mid/late`、`tyre_life_bin = young/medium/old`。 |
| ローリング特徴量 / Rolling feature | 直近数行の平均や標準偏差を使う時系列特徴量です。例：同じ driver/race/stint 内の直近3ラップの `LapTime_Delta` 平均。 |
| ターゲットエンコーディング / Target encoding | カテゴリを、そのカテゴリの過去の目的変数平均に置き換える方法です。リークを防ぐため、CV内で慎重に作る必要があります。 |
| 頻度エンコーディング / Frequency encoding | カテゴリの出現回数や頻度を特徴量にする方法です。例：`Driver_Race` が何回出たか。 |
| グループ統計量 / Group statistics | グループごとの平均、標準偏差、平均との差などです。例：`LapTime_Delta_std_by_Race_Year`。 |
| ブレンディング / Blending | 複数モデルや複数submissionの予測を重み付き平均や順位平均で混ぜることです。 |
| スタッキング / Stacking | 複数モデルのOOF予測を入力にして、2段目のモデルを学習する方法です。 |
| リーク / Leakage | 本来予測時に使えない情報が特徴量に混ざることです。強すぎる特徴量は注意して確認する必要があります。 |

### 4. EDAレポートと可視化から分かったこと

完全な自動生成EDAは **Appendix A** に入れています。`reports/figures/eda/` にある画像もすべて相対パスで参照できるようにしています。

重要な発見は以下です。

1. `TyreLife`, `LapNumber`, `Stint`, `RaceProgress` は目的変数とかなり関係が強い。
2. `Compound` と `Race` は、タイヤ戦略やサーキットごとの違いを表すため重要。
3. `Year` は分布差が大きく、validation artifact や生成データの癖に注意が必要。
4. `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation` には大きな外れ値があり、外れ値フラグやグループ平均との差が有効そう。
5. train/test/original の分布比較は、ローカルCVとLeaderboardのズレを考えるうえで重要。

**Target distribution**

![Target distribution](reports/figures/eda/target_distribution.png)

**Numeric correlation heatmap**

![Numeric correlation heatmap](reports/figures/eda/correlation_heatmap.png)

**Correlation with target**

![Correlation with target](reports/figures/eda/correlation_with_target.png)

**Compound: frequency and target rate**

![Compound: frequency and target rate](reports/figures/eda/columns/categorical_Compound.png)

**Race: frequency and target rate**

![Race: frequency and target rate](reports/figures/eda/columns/categorical_Race.png)

**Driver: frequency and target rate**

![Driver: frequency and target rate](reports/figures/eda/columns/categorical_Driver.png)

**Year: distribution and target relationship**

![Year: distribution and target relationship](reports/figures/eda/columns/numeric_Year.png)

**LapNumber: distribution and target relationship**

![LapNumber: distribution and target relationship](reports/figures/eda/columns/numeric_LapNumber.png)

**Stint: distribution and target relationship**

![Stint: distribution and target relationship](reports/figures/eda/columns/numeric_Stint.png)

**TyreLife: distribution and target relationship**

![TyreLife: distribution and target relationship](reports/figures/eda/columns/numeric_TyreLife.png)

**RaceProgress: distribution and target relationship**

![RaceProgress: distribution and target relationship](reports/figures/eda/columns/numeric_RaceProgress.png)

**LapTime Delta: distribution and target relationship**

![LapTime Delta: distribution and target relationship](reports/figures/eda/columns/numeric_LapTime_Delta.png)

**Cumulative Degradation: distribution and target relationship**

![Cumulative Degradation: distribution and target relationship](reports/figures/eda/columns/numeric_Cumulative_Degradation.png)

**Train/test/original comparison: Race**

![Train/test/original comparison: Race](reports/figures/eda/dataset_comparison/compare_Race.png)

**Train/test/original comparison: Compound**

![Train/test/original comparison: Compound](reports/figures/eda/dataset_comparison/compare_Compound.png)

### 5. 自分が最終モデルでやったこと

| 手順 | やったこと | 効いた理由 / 効かなかった理由 |
| --- | --- | --- |
| LightGBM baseline | 表形式データ向けの安定したベースラインを作った。 | LightGBMは速くて強いため、最初の基準として適していた。 |
| 基本的なドメイン特徴量 | タイヤ寿命比、レース進行率との掛け算、ラップ/時間差、劣化系特徴量を追加。 | タイヤの古さ、レースのどの時点か、ペース低下、順位状況はピット判断に直接関係するため。 |
| 外れ値フラグ | 異常なラップタイム、ラップタイム差、劣化値をフラグ化。 | 一部はtarget liftが高かったが、該当行が少なく、効果は限定的だった。 |
| AB特徴量選択 | 候補特徴量を1つずつ試し、validationが改善したものを残した。 | AIが作った特徴量を全部信じず、検証で効いたものだけを残せた。 |
| CatBoost | 選択特徴量を使ってCatBoostを学習。 | CatBoostはカテゴリ構造やルールっぽい関係をLightGBMと違う形で拾えた。 |
| LightGBM + CatBoost blend | OOF予測を使って重みを探索。 | 2モデルの誤差が完全には同じではなかったため、平均で個別のミスを減らせた。 |
| XGBoost | 3つ目のモデルとして軽量XGBoostを追加。 | OOF AUCが低く、最終的に重み0になった。 |
| CatBoost-2000 | より長めにCatBoostを学習。 | 強かったが、最良の2モデルブレンドを超えるほどではなかった。 |

ここで重要なのは、「OOFが低かった」ではなく **OOF AUCが低かった** と書くことです。OOFは検証用予測の作り方であり、低い/高いのはAUCなどの指標です。

### 6. 主な結果

| 実験 | OOF AUC | 解釈 |
| --- | ---: | --- |
| Stage-2 LightGBM baseline | 0.955756 | 基本特徴量でかなり強いベースライン。 |
| LightGBM AB-selected | 0.958394 | 特徴量選択によりLightGBMが改善。 |
| CatBoost on selected features | 0.959197 | メイン経路で最も強い単体モデル。 |
| LightGBM + CatBoost probability blend | 0.960446 | ローカルOOFで最良。 |
| XGBoost lightweight model | 0.942677 | 最終ブレンドに入れるには弱かった。 |
| Three-model ensemble | 0.960446 | XGBoostの重みが0で、実質2モデルブレンドと同じ。 |
| CatBoost-2000 follow-up | 0.960022 | 強いが、最良の2モデルブレンドには少し届かなかった。 |

### 7. AIに100個以上特徴量を作らせた実験の反省

最終モデルを作る前に、AIに100個以上の特徴量を作らせ、それらをABテストして精度が上がるものだけを残す実験をしました。しかし、結果は0.93台にとどまり、最終的にはその流れをすべて破棄して、1から作り直しました。

ここから学んだのは、「AIが役に立たない」ということではありません。学んだのは、特徴量設計は問題の発生メカニズムから始めるべきだということです。このコンペで良い特徴量は、タイヤ寿命、スティント、ラップ番号、レース進行率、コンパウンド、サーキット、順位、ペース低下、直近の劣化など、実際のピット判断につながる必要があります。

一般化した学びとしては、**特徴量は数ではなく質**です。良い特徴量とは、目的変数が発生する理由とつながっていて、OOF AUCを安定して改善する特徴量です。

### 8. 上位notebookから学んだこと

アップロードされた上位notebook差分分析では、上位者との差は単に「モデルが良かった」だけではありませんでした。文脈特徴量、グループ統計量、頻度エンコーディング、RealMLP、OOF保存、相関を見たアンサンブル、最終段階のブレンディングまで含めた、かなり完成度の高いパイプラインになっていました。

#### 8.1 レース文脈を表すカテゴリ交互作用

上位notebookでは、以下のような特徴量が作られていました。

- `Race_Year`
- `Driver_Race`
- `Driver_Compound`
- `Race_Compound`
- `Compound_Stint`
- `Race_Compound_Stint`
- `RacePhase_TyreLifeBin`
- `Compound_TyreLifeBin`

これらが重要なのは、同じ値でも文脈によって意味が変わるからです。例えば同じ `HARD` タイヤでも、Monaco と Bahrain では戦略が違います。同じ `Stint=2` でも、レース序盤なのか終盤なのかで意味が変わります。上位者はこの違いを特徴量として明示的にモデルへ渡していました。

#### 8.2 グループ統計量と標準偏差特徴量

上位CatBoost notebookでは、以下のような特徴量が使われていました。

- `Position_Change_std_by_Race_Year`
- `LapTime_Delta_diff_mean_by_Race_Year`
- `RaceProgress_diff_mean_by_Compound_Stint`
- `TyreLife_diff_mean_by_Race_Compound_Stint`
- `LapTime_Delta_std_by_Driver_Race`
- `RaceProgress_mean_by_Driver_Race`

重要なのは、「その行の値を、その文脈の中で比較する」という考え方です。例えば `LapTime_Delta = 3` だけを見ると遅く見えます。しかし、そのレース全体でラップタイム差が大きく出やすいなら、実は普通かもしれません。

標準偏差を作ることで、そのレース内のそのラップが周囲よりどれくらい速いか・遅いか、またそのグループがどれくらいブレやすいかが分かります。

具体例：

| 行 | Race | LapTime_Delta | Race平均 | Race標準偏差 | 解釈 |
| --- | --- | ---: | ---: | ---: | --- |
| A | Monaco | 3.0 | 2.5 | 0.4 | そのレース内ではかなり遅い。 |
| B | Bahrain | 3.0 | 2.9 | 2.0 | そのレース内ではそこまで異常ではない。 |

生の値は同じでも、文脈内での意味は違います。これがグループ統計量の強さです。

#### 8.3 Frequency Encoding / 頻度エンコーディング

出現頻度は自分では思いつかなかったので、かなり良い発想だと感じました。上位notebookでは、以下のような特徴量が使われていました。

- `Driver_count`
- `Driver_freq`
- `Race_freq`
- `Driver_Race_freq`
- `Race_Year_freq`
- `Compound_TyreLifeBin_freq`

頻度エンコーディングは、「そのカテゴリがどれくらい一般的か」「どれくらい信頼できるサンプル数があるか」をモデルに教える特徴量です。

例えば、あるドライバーが多くのレースに出ている場合、そのドライバーには多くの履歴データがあります。実際のF1の文脈で考えれば、頻繁に出場しているドライバーは実力やチーム体制が安定していて、戦術が固まっている可能性もあります。ただし、今回のような匿名化・synthetic dataでは、「名ドライバー」と断定するより、**出現頻度が高いカテゴリほど統計的に信頼しやすく、低頻度カテゴリほどノイズが多い** と解釈するのが安全です。

#### 8.4 ラグ・ローリング特徴量

上位notebookでは、同じ driver × race × stint の中で、直前ラップや直近数ラップの情報も特徴量にしていました。

- `Delta_lag1`
- `Delta_roll3_mean`
- `Deg_diff`
- `TyreLife_growth`
- `LapTime_lag1`
- `LapTime_diff`

ピットインは完全に突然起こるというより、タイヤが古くなり、ラップタイムが落ち、劣化が進み、戦略ウィンドウに入って発生することが多いです。ラグやローリング特徴量は、この「流れ」を捉えるために有効です。

#### 8.5 digit / string signature 特徴量

一部の上位CatBoost notebookでは、数値を丸めて文字列化したり、桁情報を特徴量にしたりしていました。

- `TyreAgeRatio_str`
- `EstimatedTotalLaps_str`
- `Cumulative_Degradation_sig_4`
- `LapTime (s)_sig_3`
- `Position_Change_int_digit_1`

これは実務分析では少し不自然に見えることもありますが、Playground系の synthetic dataset では効くことがあります。データ生成過程や丸め方の癖が残っている場合、モデルがそこを拾えるからです。ただし、過学習リスクも高いため、OOFだけでなくPublic/PrivateのズレやGroup CVでの安定性を見る必要があります。

#### 8.6 「とあるシルバーメダル獲得した上位CatBoost notebookでは、約300個の特徴量を使っていた」

とある **シルバーメダル獲得した上位CatBoost notebook** では、最終的に約 **300個の特徴量** を使っていました。最初に聞くと危険に見えますが、それらはランダムな特徴量ではありませんでした。カテゴリ交互作用、グループ統計量、頻度特徴量、ラグ/ローリング特徴量、CatBoostと相性のよいカテゴリ特徴量が体系的に作られていました。

ここでの学びは、「常に300個使えばよい」ではありません。多くの特徴量が有効になるのは、モデルが強く、early stoppingがあり、foldごとのスコアが安定していて、特徴量が文脈を説明している場合です。

### 9. RealMLP：アルゴリズム、ユースケース、強み弱み、ハイパーパラメータ

RealMLP は、PyTabKit に含まれる表形式データ向けのニューラルネット系モデルです。LightGBM、XGBoost、CatBoost は木ベースのモデルですが、RealMLP は MLP、つまり多層パーセプトロン系のモデルです。

#### 9.1 RealMLPのアルゴリズムを初心者向けに説明

普通のMLPは、たくさんの小さな計算機が層になってつながっているモデルです。

1. 入力として `TyreLife`, `RaceProgress`, `Compound`, `Race` などを受け取る。
2. 各層で、それらの値を重み付きで組み合わせる。
3. 活性化関数によって、単純な直線ではない複雑な関係を学ぶ。
4. 最後に `PitNextLap = 1` の確率を出す。

木モデルなら、例えばこういうルールを作ります。

```text
もし TyreLife > 20 かつ RaceProgress > 0.55 かつ Compound == HARD なら、ピット確率が高い
```

一方、MLPはよりなめらかな関係を学びます。

```text
タイヤ寿命、劣化、レース進行率が一緒に上がるほど、ピットリスクがだんだん上がる
```

RealMLP は、普通のMLPを表形式データで使いやすくするために、チューニング済みのデフォルト設定や学習上の工夫を入れたモデルです。そのため、自分で適当に作るMLPよりも、Kaggleのtabular dataで試しやすいです。

#### 9.2 なぜ今回RealMLPが効いた可能性があるのか

自分の最終モデルは主に木ベースでした。LightGBM、CatBoost、XGBoost は違うモデルではありますが、大きく見るとすべて gradient boosted decision trees 系です。そのため、予測の相関が高くなりがちです。

RealMLP はニューラルネット系なので、木モデルとは違う間違い方をする可能性があります。アンサンブルでは、単体性能が少し低くても、他モデルと違うミスをするモデルは価値があります。

具体例：

| 行 | 正解 | LightGBM | CatBoost | RealMLP |
| --- | ---: | ---: | ---: | ---: |
| 古いタイヤ・終盤 | 1 | 0.86 | 0.83 | 0.80 |
| 珍しいdriver/race | 1 | 0.42 | 0.45 | 0.68 |
| 変なラップタイム外れ値 | 0 | 0.75 | 0.70 | 0.40 |

RealMLPが珍しい組み合わせやなめらかな数値関係に強ければ、木モデルと混ぜたときに順位が改善する可能性があります。

#### 9.3 RealMLPのユースケース

RealMLPは次のような場面で試す価値があります。

- 表形式データである。
- データ数が中規模から大規模である。
- 連続数値特徴量が多い。
- LightGBM/CatBoostが強いが、モデル同士の予測相関が高い。
- コンペ終盤で、アンサンブルの多様性が欲しい。
- GPUが使える。
- OOF予測とtest予測を保存して、あとでブレンドやスタッキングに使える。
- 目的変数との関係が、単純なif文ルールだけでなく、なめらかな数値関係を含んでいそう。

#### 9.4 RealMLPの強み

| 強み | 説明 |
| --- | --- |
| 木モデルと違う種類のモデル | LightGBM/CatBoost/XGBoostと違う視点を持てる。 |
| なめらかな相互作用に強い | タイヤ寿命、劣化、進行率、ペース低下のような連続的な関係を拾いやすい。 |
| アンサンブルで効きやすい | 単体スコアが同程度でも、ミスの仕方が違えばblendで伸びる。 |
| チューニング済みデフォルトがある | PyTabKitのRealMLPは、最初から比較的強い設定を使いやすい。 |
| GPUと相性がよい | ニューラルネットなのでGPUで速くなる可能性がある。 |

#### 9.5 RealMLPの弱み

| 弱み | 説明 |
| --- | --- |
| 前処理に敏感 | スケーリング、欠損処理、カテゴリ処理、validation設計が大事。 |
| LightGBMより重いことがある | ニューラルネットなので学習時間や環境構築の負担が大きい。 |
| 解釈しにくい | 木モデルのfeature importanceほど直感的に説明しにくい。 |
| 過学習する可能性 | 特徴量が多い、epochが長い、validationが弱いと過学習しやすい。 |
| 環境構築が面倒な場合がある | PyTorch、PyTabKit、GPU周りで詰まりやすい。 |

#### 9.6 RealMLPで見るべきハイパーパラメータ

| ハイパーパラメータ / 設定 | 意味 | 実践的な見方 |
| --- | --- | --- |
| `learning_rate` / `lr` | 学習の一歩の大きさ。 | 大きすぎると不安定、小さすぎると遅い。重要度が高い。 |
| hidden width | 隠れ層のユニット数。 | 大きいほど複雑な関係を学べるが、過学習・時間増加のリスク。 |
| number of layers | ニューラルネットの深さ。 | 深いほど複雑になるが、学習が難しくなる。 |
| batch size | 一度に処理する行数。 | GPUでは大きめが速いことが多いが、汎化への影響もある。 |
| epochs / max epochs | 最大学習回数。 | 最大まで回すより early stopping を使う方がよい。 |
| early stopping / best epoch | validationが伸びなくなったら止める仕組み。 | tabular neural netではかなり重要。 |
| weight decay | 重みに対する正則化。 | 過学習を抑える。 |
| dropout | 学習中に一部のユニットをランダムに無効化する。 | 適度なら過学習防止になるが、強すぎると性能低下。 |
| validation fraction / fold split | validationの作り方。 | コンペ構造に合っていないとOOFが信用しにくい。 |
| random seed | 乱数。 | 不安定なら複数seed平均を試す。 |
| ensemble / `n_refit` 系 | 複数NNの平均や再学習。 | 単体RealMLPがブレる場合に有効。 |
| categorical/numerical embedding settings | カテゴリや数値を内部表現に変える設定。 | 高カーディナリティカテゴリが多いと重要。 |

実践では、まず PyTabKit の tuned-default RealMLP classifier を動かし、OOF/test予測を保存するのが良いです。その後、時間があればHPOやmulti-seed ensembleを試す流れが現実的です。

### 10. OOF予測保存、モデル間相関、重み探索

上位者は、OOF予測を「あとで使う資産」として保存していました。これはかなり大きなワークフローの差です。

各モデルごとに、最低でも以下を保存します。

```text
oof_lgbm.csv
oof_catboost.csv
oof_realmlp.csv
test_lgbm.csv
test_catboost.csv
test_realmlp.csv
fold_metrics.csv
```

そのうえで、モデル同士のOOF相関を見ます。

| モデルペア | OOF相関 | 解釈 |
| --- | ---: | --- |
| LightGBM vs CatBoost | 0.98 | かなり似ている。ブレンドの伸びは限定的だが、少しは効く可能性。 |
| LightGBM vs RealMLP | 0.90 | ある程度違う。アンサンブル多様性として価値がありそう。 |
| CatBoost vs 弱いXGBoost | 0.85 | 違いはあるが、XGBoostのAUCが低いならノイズになりやすい。 |

ポイントは、「違うモデルなら何でもよい」ではないことです。**ある程度強く、かつ他モデルと少し違う** モデルがアンサンブルに向いています。

重み探索の具体例：

```python
best_auc = -1
best_w = None
for w in np.linspace(0, 1, 101):
    blend_oof = w * oof_lgbm + (1 - w) * oof_cat
    auc = roc_auc_score(y, blend_oof)
    if auc > best_auc:
        best_auc = auc
        best_w = w

blend_test = best_w * test_lgbm + (1 - best_w) * test_cat
```

自分のプロジェクトでは、LightGBM/CatBoostの確率ブレンドで **LightGBM 0.427 / CatBoost 0.573** 付近が最良のOOF AUCになりました。

### 11. ブレンディングの具体的なやり方

#### 11.1 確率ブレンド

一番シンプルな方法です。

```text
final = 0.60 * CatBoost + 0.40 * LightGBM
```

両方のモデルの確率スケールがある程度信用できる場合に向いています。

#### 11.2 Rank blending / 順位ブレンド

AUCでは確率の絶対値より順位が重要です。そこで、予測確率を順位に変換してから平均します。

```text
Model A probabilities: 0.10, 0.90, 0.40
Model A ranks:         1,    3,    2
```

確率分布は違うけど、順位情報は有用なモデル同士を混ぜるときに効くことがあります。

#### 11.3 Rank-remap blending

Rank-remap blending は、強いsubmissionをanchorとして使い、別モデルの順位情報だけを反映する方法です。

流れは以下です。

1. 候補モデルの順位を取る。
2. 信頼しているanchor submissionの確率分布を取る。
3. anchorの値を、候補モデルの順位に合わせて並べ替える。
4. anchorとremap済み予測を少し混ぜる。

AUCコンペでは、確率分布を大きく壊さずに順位だけ少し変えられるため、有効な場合があります。

#### 11.4 OOFを使ったstacking

Stackingでは、複数モデルのOOF予測を2段目モデルの入力にします。

| 行 | LGBM OOF | CatBoost OOF | RealMLP OOF | Target |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.20 | 0.25 | 0.18 | 0 |
| 2 | 0.80 | 0.75 | 0.85 | 1 |
| 3 | 0.45 | 0.50 | 0.30 | 0 |

例えば Logistic Regression stacker を使えば、「どのモデルをどれくらい信じるか」を学習できます。ただし、必ずOOF予測を使う必要があります。学習データに対するin-sample予測を使うと過学習します。

#### 11.5 Kaggle notebook や過去コンペの使い方

現実的なKaggleの進め方としては、以下が考えられます。

1. 現在のコンペ内で高スコアの公開notebookを探す。
2. 過去の似たPlayground系・tabular系コンペを探し、アンサンブルの型を学ぶ。
3. OOF保存、rank blending、重み探索などのコード構造を参考にする。
4. 現在のコンペの公開notebook出力を使う場合は、必ずコンペルール上問題ないか確認する。
5. そのままコピーするのではなく、自分のOOFやLeaderboardで本当に価値があるか確認する。

過去コンペは、今のtest setに対するsubmissionを作るためではなく、方法論を学ぶために使うのが基本です。たとえば「AUCコンペではrank averagingが効きやすい」「OOF相関を見て重みを決める」といった型を吸収できます。

### 12. 次回やるべきこと

1. LightGBM/CatBoost baselineをすぐ作る。
2. 最初の日からOOF予測とtest予測を必ず保存する。
3. `Race_Year`, `Race_Compound_Stint`, `Driver_Race` のようなカテゴリ交互作用を早めに作る。
4. 高カーディナリティカテゴリにはfrequency encodingを入れる。
5. group mean/std/differenceで、その行が文脈内で普通か異常かを表す。
6. driver × race × stint 内でlag/rolling特徴量を作る。
7. RealMLPを終盤ではなく早めに試す。
8. ブレンド前にモデル間のOOF相関を見る。
9. probability blend、rank blend、rank-remap、stackingを試す。
10. OOF最強だけでなく、崩れにくいsafe blendも最終候補に残す。

### 13. 最終的な反省

自分の最終解法は悪くありませんでしたが、勾配ブースティング中心に考えすぎていました。上位者は、特徴量、モデル多様性、OOF管理、相関を見たアンサンブル、最終submissionのブレンディングまで、一つのパイプラインとして設計していました。

今回の一番大きな学びはこれです。

> 強いKaggle解法は、単に一つの良いモデルではなく、特徴量、検証、複数モデル、保存したOOF、慎重なブレンド、保守的な最終選択まで含めた一つの仕組みである。

---

# Appendix A. Auto-generated EDA Report / 自動生成EDAレポート

このセクションは `make_eda_report.py` によって自動生成されています。目的は、`id` を除く全カラムについて、分布・目的変数との関係・データセット間の分布差を確認することです。

## 1. Data Overview / データ概要

| dataset           |   rows |   columns |   missing_cells |   missing_pct |   duplicate_rows |
|:------------------|-------:|----------:|----------------:|--------------:|-----------------:|
| train             | 439140 |        16 |               0 |             0 |                0 |
| test              | 188165 |        15 |               0 |             0 |                0 |
| sample_submission | 188165 |         2 |               0 |             0 |                0 |

`train` はモデル学習用、`test` は提出予測用、`sample_submission` は提出形式確認用です。`original` が読み込まれている場合は、Playgroundのsynthetic dataとは別系統の元データとして、分布比較や追加学習データ候補の確認に使います。

## 2. Target Overview / 目的変数の確認

**目的変数の分布 / Target distribution**

`PitNextLap` のクラス比率を確認するグラフ。AUC評価ではクラス比率そのものより、正例をどれだけ上位に並べられるかが重要です。

![目的変数の分布 / Target distribution](figures/eda/target_distribution.png)

## 3. Column Summary / カラム一覧

| column                 | role    | dtype   |   missing |   missing_pct |   unique | sample_values                                                                                         |
|:-----------------------|:--------|:--------|----------:|--------------:|---------:|:------------------------------------------------------------------------------------------------------|
| id                     | id      | int64   |         0 |             0 |   439140 | 0, 1, 2, 3, 4                                                                                         |
| Driver                 | feature | str     |         0 |             0 |      887 | D109, D086, ZON, SPE, D019                                                                            |
| Compound               | feature | str     |         0 |             0 |        5 | HARD, MEDIUM, INTERMEDIATE, SOFT, WET                                                                 |
| Race                   | feature | str     |         0 |             0 |       26 | Canadian Grand Prix, Dutch Grand Prix, Austrian Grand Prix, Pre-Season Testing, Azerbaijan Grand Prix |
| Year                   | feature | int64   |         0 |             0 |        4 | 2022, 2025, 2023, 2024                                                                                |
| PitStop                | feature | int64   |         0 |             0 |        2 | 0, 1                                                                                                  |
| LapNumber              | feature | int64   |         0 |             0 |       78 | 50, 27, 59, 2, 26                                                                                     |
| Stint                  | feature | int64   |         0 |             0 |        8 | 2, 3, 1, 4, 5                                                                                         |
| TyreLife               | feature | float64 |         0 |             0 |       78 | 39.0, 7.0, 22.0, 2.0, 6.0                                                                             |
| Position               | feature | int64   |         0 |             0 |       20 | 8, 4, 13, 7, 2                                                                                        |
| LapTime (s)            | feature | float64 |         0 |             0 |    37719 | 78.491, 75.095, 70.945, 94.361, 107.878                                                               |
| LapTime_Delta          | feature | float64 |         0 |             0 |    57532 | -7.564000000000007, -32.617000000000004, -7.540000000000006, -7.323999999999998, 8.965000000000003    |
| Cumulative_Degradation | feature | float64 |         0 |             0 |   142701 | 21.019000000000005, -223.207, -100.529, -7.323999999999998, -14.138999999999996                       |
| RaceProgress           | feature | float64 |         0 |             0 |     1898 | 0.7142857142857143, 0.3461538461538461, 0.8194444444444444, 0.0769230769230769, 0.3611111111111111    |
| Position_Change        | feature | float64 |         0 |             0 |       37 | 5.0, -3.0, 3.0, 0.0, -9.0                                                                             |
| PitNextLap             | target  | float64 |         0 |             0 |        2 | 1.0, 0.0                                                                                              |

## 4. Correlation / 相関関係

### 4.1 Correlation with Target / 目的変数との相関

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

**数値変数同士の相関 / Numeric correlation heatmap**

数値変数同士の相関を確認します。相関が高すぎる特徴量は、似た情報を持っている可能性があります。

![数値変数同士の相関 / Numeric correlation heatmap](figures/eda/correlation_heatmap.png)

**目的変数 `PitNextLap` との相関**

各数値変数と `PitNextLap` の線形相関を確認します。ただし、非線形な関係は相関係数だけでは見えない点に注意します。

![目的変数 PitNextLap との相関](figures/eda/correlation_with_target.png)

### 4.2 Category-wise Target Rate / カテゴリ別目的変数率

カテゴリ変数や離散値について、カテゴリ別の件数と平均目的変数率を確認します。

| column   | category                  |   count |   pit_rate |
|:---------|:--------------------------|--------:|-----------:|
| Compound | HARD                      |  170518 |     0.3275 |
| Compound | SOFT                      |   38744 |     0.1935 |
| Compound | INTERMEDIATE              |   17382 |     0.1523 |
| Compound | MEDIUM                    |  211141 |     0.1011 |
| Compound | WET                       |    1355 |     0.0251 |
| Driver   | NOR                       |    1563 |     0.2834 |
| Driver   | WEB                       |    1576 |     0.2563 |
| Driver   | BUT                       |    1655 |     0.2453 |
| Driver   | TRU                       |    1613 |     0.2436 |
| Driver   | RAI                       |    1669 |     0.2427 |
| Driver   | BIA                       |    1539 |     0.2359 |
| Driver   | KOV                       |    1607 |     0.2358 |
| Driver   | KUB                       |    1650 |     0.2339 |
| Driver   | PIQ                       |    1493 |     0.2338 |
| Driver   | MAS                       |    1682 |     0.2313 |
| Driver   | CHI                       |    1505 |     0.2306 |
| Driver   | HEI                       |    1587 |     0.2294 |
| Driver   | BUE                       |    1508 |     0.2288 |
| Driver   | GLO                       |    1509 |     0.2286 |
| Driver   | BER                       |    1489 |     0.2277 |
| Driver   | ALG                       |    1527 |     0.2266 |
| Driver   | DIR                       |    1495 |     0.2254 |
| Driver   | NAK                       |    1534 |     0.2243 |
| Driver   | PET                       |    1493 |     0.223  |
| Driver   | GUT                       |    1529 |     0.2224 |
| Driver   | SUT                       |    1579 |     0.2217 |
| Driver   | D001                      |    1498 |     0.2203 |
| Driver   | FIS                       |    1651 |     0.2199 |
| Driver   | BAR                       |    1656 |     0.2198 |
| Driver   | MAL                       |    1494 |     0.2195 |
| PitStop  | 1.0                       |   59775 |     0.2478 |
| PitStop  | 0.0                       |  379365 |     0.1913 |
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
| Position | 7.0                       |   24690 |     0.1951 |
| Position | 5.0                       |   24398 |     0.1928 |
| Position | 3.0                       |   24285 |     0.1927 |
| Position | 6.0                       |   24845 |     0.1902 |
| Position | 2.0                       |   21427 |     0.1899 |
| Position | 18.0                      |   15869 |     0.1888 |
| Position | 4.0                       |   25267 |     0.1785 |
| Position | 19.0                      |   10615 |     0.1689 |
| Position | 1.0                       |   23841 |     0.1593 |
| Position | 20.0                      |    5808 |     0.1541 |
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
| Race     | Azerbaijan Grand Prix     |   12126 |     0.2146 |
| Race     | Japanese Grand Prix       |   12891 |     0.204  |
| Race     | Austrian Grand Prix       |   21223 |     0.1881 |
| Race     | Australian Grand Prix     |   18406 |     0.1816 |
| Race     | Dutch Grand Prix          |   24462 |     0.1761 |
| Race     | Qatar Grand Prix          |   13817 |     0.1756 |
| Race     | Canadian Grand Prix       |   21416 |     0.1539 |
| Race     | Abu Dhabi Grand Prix      |   16427 |     0.1505 |
| Race     | Pre-Season Testing        |   22492 |     0.1465 |
| Race     | Singapore Grand Prix      |   18960 |     0.1418 |
| Race     | British Grand Prix        |   15383 |     0.1335 |
| Race     | Italian Grand Prix        |   19854 |     0.132  |
| Race     | United States Grand Prix  |   18045 |     0.114  |
| Race     | Miami Grand Prix          |   18860 |     0.1036 |
| Race     | Mexico City Grand Prix    |   23672 |     0.0907 |
| Stint    | 2.0                       |  129536 |     0.3911 |
| Stint    | 3.0                       |   69238 |     0.2931 |
| Stint    | 4.0                       |   18903 |     0.1717 |

## 5. Per-column EDA / 各カラムの詳細EDA

以下では `id` を除く全カラムについて、型に合わせてヒストグラム、箱ひげ図、棒グラフ、円グラフ、線グラフなどを使って確認します。

### Driver

- Type: `categorical_high_cardinality`

- dtype: `str`

- Missing: 0 (0.00%)

- Unique values: 887


`Driver` はカテゴリ変数です。ユニーク数は 887 で、件数が多いカテゴリは MAS (1,682), RAI (1,669), BAR (1,656), BUT (1,655), FIS (1,651) です。 十分な件数があるカテゴリの中では、`PitNextLap`率が高い例は STR: 42.75%, BEA: 40.00%, ALO: 39.32%、低い例は D198: 13.18%, D186: 13.33%, D201: 13.37% です。

**Driver の頻度と目的変数との関係**

カテゴリカラム `Driver` について、カテゴリ頻度、カテゴリ別 `PitNextLap` 率、必要に応じて円グラフを確認します。

![Driver の頻度と目的変数との関係](figures/eda/columns/categorical_Driver.png)

### Compound

- Type: `categorical`

- dtype: `str`

- Missing: 0 (0.00%)

- Unique values: 5


`Compound` はカテゴリ変数です。ユニーク数は 5 で、件数が多いカテゴリは MEDIUM (211,141), HARD (170,518), SOFT (38,744), INTERMEDIATE (17,382), WET (1,355) です。 十分な件数があるカテゴリの中では、`PitNextLap`率が高い例は HARD: 32.75%, SOFT: 19.35%, INTERMEDIATE: 15.23%、低い例は WET: 2.51%, MEDIUM: 10.11%, INTERMEDIATE: 15.23% です。

**Compound の頻度と目的変数との関係**

カテゴリカラム `Compound` について、カテゴリ頻度、カテゴリ別 `PitNextLap` 率、必要に応じて円グラフを確認します。

![Compound の頻度と目的変数との関係](figures/eda/columns/categorical_Compound.png)

### Race

- Type: `categorical`

- dtype: `str`

- Missing: 0 (0.00%)

- Unique values: 26


`Race` はカテゴリ変数です。ユニーク数は 26 で、件数が多いカテゴリは Dutch Grand Prix (24,462), Mexico City Grand Prix (23,672), Pre-Season Testing (22,492), Hungarian Grand Prix (22,481), Monaco Grand Prix (21,539) です。 十分な件数があるカテゴリの中では、`PitNextLap`率が高い例は Chinese Grand Prix: 38.86%, Monaco Grand Prix: 35.74%, Spanish Grand Prix: 32.00%、低い例は Mexico City Grand Prix: 9.07%, Miami Grand Prix: 10.36%, United States Grand Prix: 11.40% です。

**Race の頻度と目的変数との関係**

カテゴリカラム `Race` について、カテゴリ頻度、カテゴリ別 `PitNextLap` 率、必要に応じて円グラフを確認します。

![Race の頻度と目的変数との関係](figures/eda/columns/categorical_Race.png)

### Year

- Type: `numeric_discrete_or_ordinal`

- dtype: `int64`

- Missing: 0 (0.00%)

- Unique values: 4


`Year` は数値変数です。中央値は 2024.0000、平均は 2023.5235、範囲は 2022.0000 〜 2025.0000 です。 `PitNextLap` との相関は 0.1253 です。 クラス別平均は PitNextLap=0.0: 2023.4596, PitNextLap=1.0: 2023.7811 です。

**Year の分布と目的変数との関係**

数値カラム `Year` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![Year の分布と目的変数との関係](figures/eda/columns/numeric_Year.png)

### PitStop

- Type: `numeric_discrete_or_ordinal`

- dtype: `int64`

- Missing: 0 (0.00%)

- Unique values: 2


`PitStop` は数値変数です。中央値は 0.0000、平均は 0.1361、範囲は 0.0000 〜 1.0000 です。 `PitNextLap` との相関は 0.0486 です。 クラス別平均は PitNextLap=0.0: 0.1278, PitNextLap=1.0: 0.1695 です。

**PitStop の分布と目的変数との関係**

数値カラム `PitStop` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![PitStop の分布と目的変数との関係](figures/eda/columns/numeric_PitStop.png)

### LapNumber

- Type: `numeric_continuous`

- dtype: `int64`

- Missing: 0 (0.00%)

- Unique values: 78


`LapNumber` は数値変数です。中央値は 19.0000、平均は 23.1059、範囲は 1.0000 〜 78.0000 です。 `PitNextLap` との相関は 0.2671 です。 クラス別平均は PitNextLap=0.0: 20.8487, PitNextLap=1.0: 32.1925 です。

**LapNumber の分布と目的変数との関係**

数値カラム `LapNumber` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![LapNumber の分布と目的変数との関係](figures/eda/columns/numeric_LapNumber.png)

### Stint

- Type: `numeric_discrete_or_ordinal`

- dtype: `int64`

- Missing: 0 (0.00%)

- Unique values: 8


`Stint` は数値変数です。中央値は 2.0000、平均は 1.7891、範囲は 1.0000 〜 8.0000 です。 `PitNextLap` との相関は 0.1982 です。 クラス別平均は PitNextLap=0.0: 1.6953, PitNextLap=1.0: 2.1670 です。

**Stint の分布と目的変数との関係**

数値カラム `Stint` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![Stint の分布と目的変数との関係](figures/eda/columns/numeric_Stint.png)

### TyreLife

- Type: `numeric_continuous`

- dtype: `float64`

- Missing: 0 (0.00%)

- Unique values: 78


`TyreLife` は数値変数です。中央値は 12.0000、平均は 14.1582、範囲は 1.0000 〜 77.0000 です。 `PitNextLap` との相関は 0.2735 です。 クラス別平均は PitNextLap=0.0: 12.8221, PitNextLap=1.0: 19.5369 です。

**TyreLife の分布と目的変数との関係**

数値カラム `TyreLife` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![TyreLife の分布と目的変数との関係](figures/eda/columns/numeric_TyreLife.png)

### Position

- Type: `numeric_discrete_or_ordinal`

- dtype: `int64`

- Missing: 0 (0.00%)

- Unique values: 20


`Position` は数値変数です。中央値は 10.0000、平均は 9.6303、範囲は 1.0000 〜 20.0000 です。 `PitNextLap` との相関は 0.0213 です。 クラス別平均は PitNextLap=0.0: 9.5742, PitNextLap=1.0: 9.8564 です。

**Position の分布と目的変数との関係**

数値カラム `Position` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![Position の分布と目的変数との関係](figures/eda/columns/numeric_Position.png)

### LapTime (s)

- Type: `numeric_continuous`

- dtype: `float64`

- Missing: 0 (0.00%)

- Unique values: 37,719


`LapTime (s)` は数値変数です。中央値は 90.5210、平均は 90.9487、範囲は 67.6940 〜 2507.6070 です。 `PitNextLap` との相関は -0.0341 です。 クラス別平均は PitNextLap=0.0: 91.2847, PitNextLap=1.0: 89.5961 です。

**LapTime (s) の分布と目的変数との関係**

数値カラム `LapTime (s)` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![LapTime (s) の分布と目的変数との関係](figures/eda/columns/numeric_LapTime_s.png)

### LapTime_Delta

- Type: `numeric_continuous`

- dtype: `float64`

- Missing: 0 (0.00%)

- Unique values: 57,532


`LapTime_Delta` は数値変数です。中央値は -0.2950、平均は -3.7700、範囲は -2403.8950 〜 2423.9320 です。 `PitNextLap` との相関は -0.0049 です。 クラス別平均は PitNextLap=0.0: -3.6617, PitNextLap=1.0: -4.2062 です。

**LapTime_Delta の分布と目的変数との関係**

数値カラム `LapTime_Delta` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![LapTime_Delta の分布と目的変数との関係](figures/eda/columns/numeric_LapTime_Delta.png)

### Cumulative_Degradation

- Type: `numeric_continuous`

- dtype: `float64`

- Missing: 0 (0.00%)

- Unique values: 142,701


`Cumulative_Degradation` は数値変数です。中央値は -20.9940、平均は -25.7218、範囲は -274.5640 〜 2412.0260 です。 `PitNextLap` との相関は -0.1674 です。 クラス別平均は PitNextLap=0.0: -21.1524, PitNextLap=1.0: -44.1162 です。

**Cumulative_Degradation の分布と目的変数との関係**

数値カラム `Cumulative_Degradation` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![Cumulative_Degradation の分布と目的変数との関係](figures/eda/columns/numeric_Cumulative_Degradation.png)

### RaceProgress

- Type: `numeric_continuous`

- dtype: `float64`

- Missing: 0 (0.00%)

- Unique values: 1,898


`RaceProgress` は数値変数です。中央値は 0.2692、平均は 0.3377、範囲は 0.0128 〜 1.0000 です。 `PitNextLap` との相関は 0.1855 です。 クラス別平均は PitNextLap=0.0: 0.3142, PitNextLap=1.0: 0.4319 です。

**RaceProgress の分布と目的変数との関係**

数値カラム `RaceProgress` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![RaceProgress の分布と目的変数との関係](figures/eda/columns/numeric_RaceProgress.png)

### Position_Change

- Type: `numeric_discrete_or_ordinal`

- dtype: `float64`

- Missing: 0 (0.00%)

- Unique values: 37


`Position_Change` は数値変数です。中央値は 0.0000、平均は 0.1015、範囲は -18.0000 〜 18.0000 です。 `PitNextLap` との相関は 0.0462 です。 クラス別平均は PitNextLap=0.0: 0.0092, PitNextLap=1.0: 0.4732 です。

**Position_Change の分布と目的変数との関係**

数値カラム `Position_Change` について、分布、`PitNextLap` 別の箱ひげ図、bin別の平均 `PitNextLap` を確認します。

![Position_Change の分布と目的変数との関係](figures/eda/columns/numeric_Position_Change.png)

## 6. Train/Test/Original Distribution Comparison / データセット間の分布比較

train/test/original の分布が大きく違うカラムは、Public LBとCVのズレや、外部データ追加時の悪化につながる可能性があります。

**train/test/original 分布比較: Driver**

`Driver` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: Driver](figures/eda/dataset_comparison/compare_Driver.png)

**train/test/original 分布比較: Compound**

`Compound` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: Compound](figures/eda/dataset_comparison/compare_Compound.png)

**train/test/original 分布比較: Race**

`Race` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: Race](figures/eda/dataset_comparison/compare_Race.png)

**train/test/original 分布比較: Year**

`Year` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: Year](figures/eda/dataset_comparison/compare_Year.png)

**train/test/original 分布比較: PitStop**

`PitStop` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: PitStop](figures/eda/dataset_comparison/compare_PitStop.png)

**train/test/original 分布比較: LapNumber**

`LapNumber` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: LapNumber](figures/eda/dataset_comparison/compare_LapNumber.png)

**train/test/original 分布比較: Stint**

`Stint` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: Stint](figures/eda/dataset_comparison/compare_Stint.png)

**train/test/original 分布比較: TyreLife**

`TyreLife` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: TyreLife](figures/eda/dataset_comparison/compare_TyreLife.png)

**train/test/original 分布比較: Position**

`Position` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: Position](figures/eda/dataset_comparison/compare_Position.png)

**train/test/original 分布比較: LapTime (s)**

`LapTime (s)` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: LapTime (s)](figures/eda/dataset_comparison/compare_LapTime_s.png)

**train/test/original 分布比較: LapTime_Delta**

`LapTime_Delta` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: LapTime_Delta](figures/eda/dataset_comparison/compare_LapTime_Delta.png)

**train/test/original 分布比較: Cumulative_Degradation**

`Cumulative_Degradation` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: Cumulative_Degradation](figures/eda/dataset_comparison/compare_Cumulative_Degradation.png)

**train/test/original 分布比較: RaceProgress**

`RaceProgress` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: RaceProgress](figures/eda/dataset_comparison/compare_RaceProgress.png)

**train/test/original 分布比較: Position_Change**

`Position_Change` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。

![train/test/original 分布比較: Position_Change](figures/eda/dataset_comparison/compare_Position_Change.png)

## 7. Notes for Feature Engineering / 特徴量作成への示唆

EDAで重要なのは、単にグラフを見ることではなく、`なぜ次のラップでピットするのか` という発生メカニズムに結びつけることです。たとえば、TyreLife、LapNumber、RaceProgress、Stint、Compound、Race、Position系の変数は、ピット戦略の理由と直接つながりやすい候補です。一方で、相関が低い変数でも、LightGBMやCatBoostのような木系モデルでは非線形な条件分岐として効く場合があります。

---

## References / 参考資料

- PyTabKit GitHub repository. Used for the general description that PyTabKit provides RealMLP variants with tuned defaults, HPO, and ensembling interfaces.
- PyTabKit documentation examples. Used for the practical idea of fitting/refitting RealMLP and saving validation-based settings.
- Holzmüller, D. et al. (2024), *Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data*. Used for the explanation that RealMLP is an improved MLP for tabular data with tuned defaults and a favorable time/accuracy tradeoff in benchmark settings.
- Uploaded project files: `REPORT.md`, `reports.zip`, and `f1_pit_prediction_reflection_ja(1).md`.
