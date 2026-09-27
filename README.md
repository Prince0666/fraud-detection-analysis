# Fraud Detection — Z-Score & Cost-Benefit Threshold Analysis

**Author:** Prince Pal | [GitHub](https://github.com/Prince0666) | [LinkedIn](https://linkedin.com/in/prince-pal-311537311)

## Business Problem

A fintech processing thousands of transactions daily needs to flag potential fraud without either (a) missing too much fraud or (b) blocking too many genuine customers. This project builds and evaluates a z-score-based fraud flag, measures its precision-recall trade-off, and finds the threshold that minimizes total business cost — not just the "most accurate" one.

## Dataset

10,000 transactions across 2,500 customers, with a realistic 1.70% fraud rate (matches real-world card-fraud base rates).

**Columns:** transaction_id, customer_id, transaction_date, transaction_hour, merchant_category, transaction_amount, is_foreign, distance_from_home_km, account_age_days, is_fraud

## Tools Used

SQL (SQLite) for exploratory querying · Python (Pandas, NumPy) for z-score computation, precision-recall evaluation, and cost-benefit analysis

---

## Part 1 — SQL Analysis

Full query-by-query breakdown: see `fraud_project_findings.md`.

| # | Question | Finding |
|---|---|---|
| 1 | Overall fraud rate | 1.70% (highly imbalanced) |
| 2 | Fraud rate by merchant category | Jewelry 14.91% (~9x average) · Electronics 3.12% · Travel 2.93% · rest under 1% |
| 3 | Fraud rate — foreign vs domestic | Domestic 1.43% · Foreign 5.62% (~4x) |
| 4 | Fraud rate — late night (1-4 AM) vs rest of day | Late night 3.54% (n=311) · Rest of day 1.64% (n=9,689), ~2.15x |
| 5 | Avg. transaction amount — fraud vs non-fraud | Non-fraud ₹195.11 · Fraud ₹1,014.40 (~5.2x) |

---

## Part 2 — Z-Score Outlier Detection & Evaluation (Python)

### 2.1 Z-Score Calculation
```python
import pandas as pd

df = pd.read_csv('fraud_detection_transactions.csv')

mean_amount = df['transaction_amount'].mean()
std_amount = df['transaction_amount'].std()
df['z_score'] = (df['transaction_amount'] - mean_amount) / std_amount
```

### 2.2 Flagging & Confusion Matrix (threshold z > 2)
```python
df['flagged_suspicious'] = (df['z_score'] > 2).astype(int)
print(pd.crosstab(df['flagged_suspicious'], df['is_fraud']))
```

| | Not Fraud | Fraud |
|---|---|---|
| Not Flagged | 9,464 | 95 |
| Flagged | 366 | 75 |

**Precision: 17.0% · Recall: 44.1%** — amount alone is a weak standalone signal; 83% of flags at this threshold are false alarms.

### 2.3 Precision-Recall Trade-off Across Thresholds
```python
thresholds = [1.5, 2, 2.5, 3, 3.5]
for t in thresholds:
    df['flagged'] = (df['z_score'] > t).astype(int)
    TP = ((df['flagged']==1) & (df['is_fraud']==1)).sum()
    FP = ((df['flagged']==1) & (df['is_fraud']==0)).sum()
    FN = ((df['flagged']==0) & (df['is_fraud']==1)).sum()
    precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    print(f"Threshold {t}: Precision={precision:.3f}, Recall={recall:.3f}, Flagged={df['flagged'].sum()}")
```

| Threshold | Precision | Recall | Flagged |
|---|---|---|---|
| 1.5 | 13.0% | 50.0% | 654 |
| 2.0 | 17.0% | 44.1% | 441 |
| 2.5 | 21.5% | 37.6% | 297 |
| 3.0 | 29.0% | 31.8% | 186 |
| 3.5 | 38.3% | 28.8% | 128 |

Stricter thresholds trade recall for precision — a textbook precision-recall trade-off, and no threshold is "correct" without attaching a real cost to each error type.

### 2.4 Cost-Benefit Analysis
```python
avg_fraud_loss = 1014.40   # cost of a missed fraud (false negative)
review_cost = 50           # cost of a false alarm (false positive)

for t in thresholds:
    df['flagged'] = (df['z_score'] > t).astype(int)
    FN = ((df['flagged']==0) & (df['is_fraud']==1)).sum()
    FP = ((df['flagged']==1) & (df['is_fraud']==0)).sum()
    total_cost = (FN * avg_fraud_loss) + (FP * review_cost)
    print(f"Threshold {t}: FN={FN}, FP={FP}, Total Cost=₹{total_cost:,.0f}")
```

| Threshold | FN | FP | Total Cost |
|---|---|---|---|
| 1.5 | 85 | 569 | ₹1,14,674 |
| **2.0** | 95 | 366 | **₹1,14,668** |
| 2.5 | 106 | 233 | ₹1,19,176 |
| 3.0 | 116 | 132 | ₹1,24,270 |
| 3.5 | 121 | 79 | ₹1,26,692 |

**Threshold 2.0 minimizes total cost** — because a missed fraud (₹1,014) costs ~20x more than a false alarm (₹50), a looser threshold that catches more fraud is cheaper overall, even with more false positives. This is the central argument for cost-based thresholding over optimizing precision/recall alone.

---

## Key Insights

1. Amount-based z-score alone is a weak fraud signal (17-38% precision depending on threshold) — real systems need to combine it with categorical signals.
2. Category, geography (foreign), and time-of-day are strong standalone signals: Jewelry (~9x baseline), foreign transactions (~4x), late night (~2.15x).
3. The "best" threshold depends entirely on the relative cost of false positives vs false negatives — a looser threshold can be cheaper overall even with more false alarms, if missed fraud is expensive enough.

## Monitoring Framework & Recommendation

**Base flag:** z-score > 2.0 on transaction_amount
**Layered risk points:** +1 for foreign transactions, +1 for Jewelry/Electronics/Travel category, +1 for late-night (1-4 AM) transactions
**Escalation rule:** z-score > 2 AND 2+ risk points → auto-hold for verification; z-score > 2 with 0-1 risk points → soft flag for batch review
**Review cadence:** Re-run this threshold/cost analysis monthly as fraud patterns shift; track total ₹ cost as the primary KPI, not precision/recall in isolation

---

## Files in this project
- `fraud_detection_transactions.csv` — dataset
- `fraud_project_findings.md` — full SQL + Python query-by-query log
- `fraud_eda.ipynb` — Python notebook (z-score, precision-recall, cost-benefit)
