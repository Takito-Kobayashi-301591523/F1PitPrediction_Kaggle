# EDA Report / 探索的データ分析

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
