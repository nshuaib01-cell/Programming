# Web Scraping and Drone Warning Classification

A two-part Python assignment covering web scraping and decision tree classification, built in a Jupyter Notebook.

---

## Overview

### Part 1 — Publication Web Scraping
Scrapes a researcher's publication page to extract and organise academic publication data. The data is stored in a structured DataFrame and sorted in multiple ways for analysis.

**Data extracted per publication:**
- Title
- Year
- Venue (journal or conference)
- Authors
- Citations
- Impact Factor

**Sorting methods applied:**
- By publication year (descending)
- By number of authors (descending)
- By publication title (alphabetical)
- By impact factor (descending)

The results are exported to an Excel file (`PublicationInfo.xlsx`).

---

### Part 2 — Drone Battery Warning Classification
Builds decision tree classifiers to predict drone battery warning states from telemetry data, comparing two feature selection methods and assessing the impact of outlier removal.

**Dataset:** `drone.csv` — drone telemetry readings including voltage, current, battery remaining, and a warning label (0–3).

**Pipeline:**
- Train/test split with stratification on the warning variable
- Feature selection using two methods:
  - **SelectKBest** (univariate, Pearson correlation) — selects `current_a`, `current_filtered_a`, `scale`
  - **RFE** (Recursive Feature Elimination with Logistic Regression) — selects `voltage_filtered_v`, `current_a`, `remaining`
- Multicollinearity check via correlation heatmap
- Feature binning into 5 equal-width bins
- Outlier removal per warning class using IQR bounds
- Decision tree classifiers (max depth 3) fitted on:
  - Binned data (with outliers)
  - Binned data (outliers removed)
- Model evaluation using accuracy score, confusion matrix, and classification report

---

## Requirements

```bash
pip install requests beautifulsoup4 pandas openpyxl scikit-learn matplotlib seaborn
```

---

## File Structure

```
├── Assignment.ipynb    # Full notebook (Part 1 and Part 2)
├── drone.csv           # Drone telemetry dataset (required for Part 2)
└── README.md
```

---

## Notes

- Part 1 scrapes from a university assignment URL and requires network access
- `verify=False` is used in the HTTP request due to the assignment environment — SSL verification is disabled intentionally
- The logistic regression in Part 2 may produce a `ConvergenceWarning` — increasing `max_iter` resolves this
