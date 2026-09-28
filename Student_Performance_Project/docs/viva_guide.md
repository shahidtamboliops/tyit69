# Viva Guide

## A. Viva questions and simple answers
1. **What is your project about?** Predicting a student's final maths grade (G3) from background and study-related factors using supervised learning.
2. **Why supervised learning?** Because every student has a known label (G3) to learn from. The model learns the pattern between inputs and this known output.
3. **Why regression, not classification?** G3 is a number from 0 to 20, not a category. (Classification would need pass/fail labels.)
4. **Where is the data from and is it primary or secondary?** UCI repository; secondary data (collected earlier by others - school reports and questionnaires in Portugal).
5. **What is the target and what are the independent variables?** Target: G3. Independent: all other columns (age, studytime, failures, absences, etc.).
6. **How did you handle missing values / duplicates?** I checked with `isnull().sum()` and `duplicated().sum()` and report what I found; I would fill or drop if present.
7. **How did you handle categorical variables?** One-hot encoding (`get_dummies`, `drop_first=True`) because models need numbers.
8. **What is an outlier and what did you do?** A value far from the rest (IQR rule). I kept them since they are real students, not errors; I say so as a limitation for Linear Regression.
9. **Why two scenarios?** G1 and G2 are earlier grades and strongly predict G3. Scenario B removes them to see whether background factors alone can predict, which is what a teacher would want early in the year.
10. **Why a baseline model?** It always predicts the mean grade. If my model can't beat it, it has learned nothing.
11. **Why Linear Regression?** Simple, fast, coefficients are easy to explain. **Why Random Forest?** Handles non-linear patterns and interactions, robust to outliers, gives feature importance.
12. **Why split 80:20?** To test on students the model has never seen, giving an honest performance estimate.
13. **What is random_state=42?** A fixed seed so the split and forest give the same results every run.
14. **What is overfitting?** Memorising training data and performing poorly on new data. Detected when train performance is much better than test performance; cross-validation helps check.
15. **Why StandardScaler in a pipeline?** So coefficients are comparable across features, and scaling is learned only from training data (avoids data leakage).
16. **What is data leakage?** Information from the test set influencing training. I avoid it by splitting first and fitting the scaler inside the pipeline.
17. **Explain MAE, MSE, RMSE, R2.** See section D.
18. **Which model is best?** Whichever has the higher test R2 and lower errors in YOUR output. Quote your own numbers.
19. **Does high correlation mean causation?** No. Studytime correlating with grade does not prove it causes it.
20. **What are the limitations?** See section E.

## B. Explanation of each important code section
| Section | What it does | Why |
|---|---|---|
| Imports | Loads pandas, numpy, matplotlib, seaborn, sklearn | Tools for data, plots, models |
| `pd.read_csv(sep=";")` | Reads the file | UCI file is semicolon separated |
| `df.shape/head()/dtypes` | Size, sample, column types | First inspection |
| `isnull().sum()`, `duplicated().sum()` | Data quality checks | Bad data gives bad models |
| IQR outlier loop | Q1 - 1.5*IQR, Q3 + 1.5*IQR limits, counts values outside | Detect unusual values |
| Feature engineering | Averages of alcohol use and parents' education | Fewer overlapping columns, simpler |
| `pd.get_dummies` | Text categories -> 0/1 columns | Models need numbers |
| `agg(["mean","median","std","min","max"])` | Descriptive statistics | Summarise each variable |
| `histplot`, `boxplot`, `scatterplot`, `heatmap` | Distribution, outliers, relationships, correlation | Understand data visually |
| `train_test_split` | 80% train, 20% test | Fair evaluation |
| `Pipeline([StandardScaler, LinearRegression])` | Scale then fit, in one object | Prevents leakage |
| `DummyRegressor` | Mean baseline | Minimum standard to beat |
| `RandomForestRegressor(n_estimators=100)` | 100 trees averaged | Stronger, non-linear model |
| `evaluate()` | Fit, predict, compute MAE/MSE/RMSE/R2, 5-fold CV | Reusable evaluation |
| Coefficients / `feature_importances_` | Shows which features matter | Interpretation |
| `interpret()` | Prints results in plain words using actual numbers | Nothing is hard-coded |

## C. Statistical concepts used
- **Mean**: average. **Median**: middle value (not affected by extremes). **Std deviation**: typical spread around the mean. **Min/Max**: range limits.
- **Skewness**: asymmetry of a distribution (long tail left/right).
- **Quartiles and IQR**: Q1 (25%), Q3 (75%), IQR = Q3 - Q1; used in the outlier rule and boxplots.
- **Outlier**: value outside Q1-1.5*IQR .. Q3+1.5*IQR.
- **Pearson correlation (r)**: -1 to +1, strength and direction of a *linear* relationship.
- **Distribution**: how values are spread (histogram, KDE curve).
- **Residual**: actual minus predicted; good models have residuals centred near 0.
- **Regression**: y = b0 + b1x1 + b2x2 + ... ; coefficients show the change in G3 for a one-unit change (in standardised units here) while others stay constant.
- **Sampling/train-test logic and cross-validation**: estimating performance on unseen data.

## D. Machine learning concepts used
- **Supervised learning**: learning from labelled examples. **Regression**: predicting a number.
- **Features (X) and target (y)**.
- **One-hot encoding**, **feature scaling (standardisation)**: (x - mean) / std.
- **Train/test split**, **random seed**.
- **Linear Regression**: fits the best straight-line/plane by minimising squared errors.
- **Random Forest**: many decision trees, each trained on a random sample; predictions averaged.
- **Baseline model**.
- **Overfitting / underfitting**, **data leakage**.
- **5-fold cross-validation**: train data split into 5 parts; train on 4, validate on 1, repeat; average the score.
- **Metrics in simple language**
  - **MAE**: average size of the mistake in marks. "On average I'm off by X marks." Easiest to explain.
  - **MSE**: average of squared mistakes; big mistakes count a lot; units are marks².
  - **RMSE**: square root of MSE, back in marks; larger than MAE when there are some big misses.
  - **R2**: share of grade variation the model explains. 1 = perfect, 0 = same as guessing the average, negative = worse than guessing the average.
- **Feature importance (RF)**: how much each feature helped the trees reduce error.

## E. Limitations of the project
- Small dataset (a few hundred students) from two Portuguese schools; results may not generalise to other countries or courses.
- Only the Maths course is used; Portuguese-course file not analysed.
- G1 and G2 dominate prediction in Scenario A; scenario B is harder and usually less accurate.
- Correlation, not causation; self-reported survey answers (alcohol, health, study time) may be inaccurate.
- Outliers (e.g. G3 = 0) kept; they can distort Linear Regression.
- No hyperparameter tuning; a single random split plus CV.
- Predictions are not clipped to the 0-20 range.
- Ethical caution: predictions must support students, not label them.

## F. Future scope
- Add the Portuguese-course dataset and compare.
- Convert to classification (pass if G3 >= 10) and use accuracy, precision, recall, F1, confusion matrix.
- Hyperparameter tuning (GridSearchCV), try Gradient Boosting, Decision Tree, Ridge/Lasso.
- Handle G3 = 0 cases separately.
- Use SHAP or permutation importance for deeper explanation.
- Build a simple web app (Streamlit) to enter student details and get an early-warning prediction.
- Validate on data from other schools.
