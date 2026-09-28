# Student Performance Prediction Using Supervised Learning
Individual Mini Project - B.Sc. IT | Python | Regression

## Folder structure
```
Student_Performance_Project/
├── Student_Performance_Project.ipynb <- Jupyter Notebook (main, with explanations)
├── student_performance_project.py   <- same project as a plain Python script
├── requirements.txt
├── README.md                        <- this file
├── data/
│   └── student-mat.csv              <- YOU must download and place it here
├── outputs/                         <- graphs (.png) and result tables (.csv) appear here
└── docs/
    ├── report_structure.md          <- template for your written report
    └── viva_guide.md                <- viva Q&A, code, statistics, ML concepts, limitations, future scope
```

## Step 1: Get the dataset
1. Open https://archive.ics.uci.edu/dataset/320/student+performance
2. Download and unzip. Copy **student-mat.csv** (Maths course) into the `data/` folder.
3. The file uses `;` as separator (the script handles this). Read the licence/citation on the page.

## Step 2: Install libraries
```
pip install -r requirements.txt
```

## Step 3: Run
**Notebook (recommended):** open a terminal in the project folder, run `jupyter notebook`, open
`Student_Performance_Project.ipynb` and choose *Run > Run All Cells*. (VS Code also works: open the
folder and the notebook.) The notebook must be opened from the project folder so `data/student-mat.csv` is found.

**Google Colab:** upload the notebook, then upload `student-mat.csv` into a folder named `data`
(Files panel > new folder `data` > upload), and Run all.

**Script:** `python student_performance_project.py` (graphs pop up; set `SHOW_PLOTS=0` to only save them).

## What the script does
Problem statement -> load and inspect data -> preprocessing (types, missing, duplicates,
outliers, feature engineering, encoding) -> EDA (descriptive stats, distributions,
correlation, relationships) -> train/test split -> Baseline, Linear Regression, Random
Forest -> evaluation (MAE, MSE, RMSE, R2, 5-fold CV R2) -> interpretation.
It runs two scenarios: A (with G1, G2) and B (without G1, G2).

## Important honesty note
The script was checked to run correctly on a dummy file with the same columns, but
NO real results were produced here. All numbers in your report must come from YOUR run
on the real dataset. Copy them from the console/outputs folder, do not invent or guess.

Citation: P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student
Performance." Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008).
