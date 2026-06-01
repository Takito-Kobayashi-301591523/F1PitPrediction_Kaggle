# Kaggle Notebook Guide For F1 Pit Stop Prediction

This document is written in both English and Japanese. English appears first, followed by Japanese translation.

## English

### Purpose

This file is a draft guide for turning the local project into a readable Kaggle Notebook. The notebook should explain not only the final score, but also why each step was tried, what improved the score, and what did not help.

### Recommended Notebook Story

1. State the task: predict whether a car will pit on the next lap.
2. Load `train.csv`, `test.csv`, and `sample_submission.csv`.
3. Explain every column in plain language.
4. Run basic EDA: target rate, compound frequency, race/year/stint effects, and numeric distributions.
5. Build a simple validation setup with AUC.
6. Train a LightGBM baseline.
7. Add carefully chosen features.
8. Use feature selection to keep useful features and drop noisy ones.
9. Train CatBoost on the same selected features.
10. Blend LightGBM and CatBoost probabilities.
11. Explain failed or weaker attempts: over-pruning, removing pit-stop signals, rank blending, and XGBoost.
12. Create `submission.csv`.
13. End with lessons learned.

### Notebook Cell Outline

#### Cell 1: Imports

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from catboost import CatBoostClassifier
```

Reason: these libraries cover data handling, validation, LightGBM, CatBoost, and AUC scoring.

#### Cell 2: Load Data

```python
DATA_DIR = "/kaggle/input/playground-series-s6e5"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(train.shape, test.shape, sample.shape)
train.head()
```

Reason: keep file loading simple and reproducible inside Kaggle.

#### Cell 3: Target Summary

```python
target = "PitNextLap"
print(train[target].value_counts())
print("positive rate:", train[target].mean())
```

Expected explanation: the positive rate is about 19.90%, so AUC is more informative than simple accuracy.

#### Cell 4: Basic EDA Tables

```python
for col in ["Compound", "Race", "Year", "Stint", "PitStop"]:
    display(
        train.groupby(col)[target]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .head(20)
    )
```

Expected explanation: tyre compound, race, year, stint, and pit-stop timing all show strong differences in next-lap pit rate.

#### Cell 5: Numeric Feature Summary

```python
num_cols = [
    "LapNumber", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change"
]

display(train[num_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T)
```

Expected explanation: lap time and degradation columns contain large outliers, so robust feature engineering is important.

#### Cell 6: Feature Engineering

```python
def add_features(train_df, test_df):
    all_df = pd.concat(
        [
            train_df.assign(_is_train=1),
            test_df.assign(_is_train=0),
        ],
        axis=0,
        ignore_index=True,
    )

    df = all_df.copy()
    df["LapTime_s"] = df["LapTime (s)"]

    # Race-level average lap time is computed without using the target.
    df["AveLapTime"] = df.groupby("Race")["LapTime_s"].transform("mean")

    # Compact strategy features used in the strongest local modeling path.
    df["tyre_life_ratio"] = df["TyreLife"] / (df["LapNumber"] + 1)
    df["progress_x_tyre"] = df["RaceProgress"] * df["TyreLife"]
    df["LapNumber_minus_TyreLife"] = df["LapNumber"] - df["TyreLife"]
    df["progress_x_degradation"] = df["RaceProgress"] * df["Cumulative_Degradation"]
    df["lap_per_progress"] = df["LapNumber"] / (df["RaceProgress"] + 1e-6)

    # In the local scripts this feature was supported by richer metadata.
    # In a compact notebook, this is a simple total-distance proxy.
    df["estimated_total_distance_km"] = df["lap_per_progress"]

    train_fe = df[df["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
    test_fe = df[df["_is_train"] == 0].drop(columns=["_is_train"]).reset_index(drop=True)
    return train_fe, test_fe

train_fe, test_fe = add_features(train, test)
```

Reason: the best local path used a compact set of strategy-related features rather than every possible engineered feature. The important idea is not that these exact formulas are perfect, but that they describe pit timing: tyre age, race progress, pace loss, and expected race length.

For reference, the local compact idea is equivalent to this simpler single-dataframe pattern:

```python
def add_features_single(df):
    df = df.copy()
    df["LapTime_s"] = df["LapTime (s)"]
    df["tyre_life_ratio"] = df["TyreLife"] / (df["LapNumber"] + 1)
    df["progress_x_tyre"] = df["RaceProgress"] * df["TyreLife"]
    df["LapNumber_minus_TyreLife"] = df["LapNumber"] - df["TyreLife"]
    df["progress_x_degradation"] = df["RaceProgress"] * df["Cumulative_Degradation"]
    df["lap_per_progress"] = df["LapNumber"] / (df["RaceProgress"] + 1e-6)
    df["estimated_total_distance_km"] = df["lap_per_progress"]
    return df
```

#### Cell 7: Selected Features

```python
features = [
    "Driver", "Compound", "Race", "Year", "PitStop",
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime_s",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "tyre_life_ratio", "progress_x_tyre",
    "LapNumber_minus_TyreLife", "AveLapTime",
    "progress_x_degradation", "lap_per_progress",
    "estimated_total_distance_km",
]
```

Reason: this mirrors the 21-feature selected set from the best LightGBM/CatBoost path. If `AveLapTime` is not available in a compact Kaggle version, compute it from race-level average lap time or remove it and revalidate.

#### Cell 8: Encode Categorical Columns

```python
cat_cols = ["Driver", "Compound", "Race"]

for col in cat_cols:
    all_values = pd.concat([train_fe[col], test_fe[col]], axis=0).astype("category")
    categories = all_values.cat.categories
    train_fe[col] = pd.Categorical(train_fe[col], categories=categories).codes
    test_fe[col] = pd.Categorical(test_fe[col], categories=categories).codes
```

Reason: tree models need categorical strings converted into numeric or categorical representations.

#### Cell 9: LightGBM Cross-Validation

```python
X = train_fe[features]
y = train_fe[target]
X_test = test_fe[features]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(train_fe))
test_lgb = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
    )

    oof_lgb[va_idx] = model.predict_proba(X_va)[:, 1]
    test_lgb += model.predict_proba(X_test)[:, 1] / skf.n_splits

print("LightGBM OOF AUC:", roc_auc_score(y, oof_lgb))
```

Expected explanation: the local selected-feature LightGBM scored about 0.958 OOF AUC.

#### Cell 10: CatBoost Cross-Validation

```python
oof_cat = np.zeros(len(train_fe))
test_cat = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=150)
    oof_cat[va_idx] = model.predict_proba(X_va)[:, 1]
    test_cat += model.predict_proba(X_test)[:, 1] / skf.n_splits

print("CatBoost OOF AUC:", roc_auc_score(y, oof_cat))
```

Expected explanation: CatBoost was stronger as a single model because it captured categorical/rule-like structure differently.

#### Cell 11: Blend

```python
w_lgb = 0.427
w_cat = 0.573

oof_blend = w_lgb * oof_lgb + w_cat * oof_cat
test_blend = w_lgb * test_lgb + w_cat * test_cat

print("Blend OOF AUC:", roc_auc_score(y, oof_blend))
```

Expected explanation: the blend improved because the two models were strong and not perfectly identical.

#### Cell 12: Submission

```python
submission = sample.copy()
submission[target] = test_blend
submission.to_csv("submission.csv", index=False)
submission.head()
```

Reason: Kaggle expects a file with `id` and predicted `PitNextLap`.

### What To Say About Failed Attempts

| Attempt | What happened | Notebook explanation |
| --- | --- | --- |
| Too many engineered features | Some early gains, but later broad feature sets were not always better. | More features can add noise and overfit validation quirks. |
| Importance-based pruning | Worsened some experiments. | A feature can look weak alone but help in combination with others. |
| Dropping pit-stop signals | Worsened strongly. | Timing-related columns were important for this target. |
| XGBoost third model | OOF was much lower and final weight was 0.0. | Diversity only helps when the extra model is strong enough. |
| Rank blending | Slightly worse than probability blending. | The probabilities were useful enough that keeping calibration helped. |

## Japanese Translation

### 目的

このファイルは、ローカルで行った作業を Kaggle Notebook として読みやすく書き直すための下書きです。最終スコアだけではなく、なぜその処理をしたのか、何がスコアを改善したのか、何が効かなかったのかを説明することを目的にしています。

### Notebook のおすすめ構成

1. 目的を書く。次のラップでピットインするかを予測する問題である。
2. `train.csv`, `test.csv`, `sample_submission.csv` を読み込む。
3. 各列の意味をやさしく説明する。
4. EDA を行う。目的変数の比率、コンパウンド頻度、Race/Year/Stint の傾向、数値列の分布を見る。
5. AUC を使った検証方法を作る。
6. LightGBM のベースラインを学習する。
7. 意味のある特徴量を追加する。
8. 特徴量選択で有用な特徴量を残し、ノイズになりそうなものを落とす。
9. 同じ特徴量で CatBoost を学習する。
10. LightGBM と CatBoost の確率をブレンドする。
11. うまくいかなかった試行も説明する。過剰な pruning、pit-stop シグナル削除、順位ブレンド、XGBoost など。
12. `submission.csv` を作る。
13. 最後に学びをまとめる。

### コードセルの流れ

上の English セクションのコードを Kaggle Notebook のセルとして順番に置けば、読みやすい notebook になります。各セルのあとに、以下のような説明を短く書くと初見の人にも伝わります。

| セル | 説明すること |
| --- | --- |
| Imports | pandas/numpy はデータ処理、LightGBM/CatBoost はモデル、AUC は評価指標。 |
| Load Data | Kaggle の input からデータを読み込む。train には正解、test には正解がない。 |
| Target Summary | `PitNextLap = 1` は約 19.90%。正解率だけでは不十分なので AUC を使う。 |
| Basic EDA | コンパウンド、レース、年、スティントでピット率が大きく違う。 |
| Numeric Summary | ラップタイムや劣化には外れ値があり、特徴量作成で注意が必要。 |
| Feature Engineering | タイヤ寿命、レース進行率、劣化、ラップ差から戦略に関係する特徴量を作る。 |
| LightGBM | 速くて強い表形式データ向けモデル。最初の強い基準になる。 |
| CatBoost | カテゴリ構造を違う形で拾えるため、LightGBM と組み合わせる価値がある。 |
| Blend | 2つの良いモデルを重み付き平均し、個別モデルの誤差を減らす。 |
| Submission | `id` と `PitNextLap` の2列で提出ファイルを作る。 |

### うまくいかなかったことも書く理由

良い notebook は、成功した手法だけでなく、失敗した手法も説明します。今回なら「特徴量を増やせば必ず良くなるわけではない」「弱いモデルを混ぜても良くならない」「重要度だけで pruning すると組み合わせで効く特徴量を落とす可能性がある」という学びが重要です。
