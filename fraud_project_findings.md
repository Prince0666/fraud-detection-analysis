# Fraud Detection — Z-Score & Precision-Recall Analysis — Findings Log

**Dataset:** `fraud_detection_transactions.csv` — 10,000 transactions, 2,500 unique customers

**Tools used:** SQL (SQLite), Python (Pandas, NumPy, Matplotlib/Seaborn)

---

## Q1. Overall fraud rate

**Query:**
```sql
SELECT count(*), avg(is_fraud) * 100.0 as fraud_percentage
FROM fraud_detection_transactions
```

**Result:** 10,000 transactions, fraud rate = **1.70%**

**Insight:** Highly imbalanced dataset (98.3% legitimate vs 1.7% fraud) — this class imbalance is exactly why precision-recall matters more than plain accuracy for this problem (a model predicting "no fraud" always would still be 98.3% "accurate" but useless).

---

## Q2. Fraud rate by merchant category

**Query:**
```sql
SELECT merchant_categor, avg(is_fraud) * 100.0 as merchant_percentage
from fraud_detection_transactions
group by merchant_categor
```

**Result:**
| Category | Fraud Rate |
|---|---|
| Jewelry | 14.91% |
| Electronics | 3.12% |
| Travel | 2.93% |
| Dining | 0.97% |
| Online Retail | 0.69% |
| ATM Withdrawal | 0.69% |
| Grocery | 0.44% |
| Fuel | 0.33% |

**Insight:** Jewelry's fraud rate is ~9x the overall average — high-value, resellable items are a common fraud target in the real world too. Electronics and Travel (also high-ticket categories) are the next highest. Supports a category-based risk tiering approach.

---

## Q3. Fraud rate — foreign vs domestic transactions

**Query:**
```sql
SELECT is_foreign, avg(is_fraud) * 100.0 as foreign_transaction
from fraud_detection_transactions
group by is_foreign
```

**Result:** Domestic (0) = 1.43% · Foreign (1) = 5.62%

**Insight:** Foreign transactions carry ~4x the fraud rate of domestic ones — a well-known real-world signal, which is why most banks require extra verification (OTP, confirmation) on foreign transactions.

---

## Q4. Fraud rate by transaction hour (late night vs rest of day)

**Query:**
```sql
select 
case 
    when transaction_hour BETWEEN 1 and 4 THEN 'Late Night (1-4 Am)'
    Else 'Rest_of_Day'
    End as time_bucket,
    avg(is_fraud) * 100.0 as fraud_percentage, 
    count(*) as txn_count
    From fraud_detection_transactions
    GROUP by time_bucket
```

**Result:** Late Night (1-4 AM) = 3.54% (n=311) · Rest of Day = 1.64% (n=9,689)

**Insight:** Late-night transactions carry ~2.15x the fraud rate, and it's a low-volume window (only 3% of all transactions) — makes it a cheap, high-value flag to add to a monitoring rule.

---

## Q5. Average transaction amount — fraud vs non-fraud

**Query:**
```sql
select is_fraud, avg(transaction_amou) as avg_amount
from fraud_detection_transactions
group by is_fraud
```

**Result:** Non-fraud = ₹195.11 · Fraud = ₹1,014.40

**Insight:** Fraudulent transactions average ~5.2x the amount of legitimate ones — the strongest single signal found so far. This is the basis for z-score outlier detection in Part 2, since z-score on transaction_amount should catch a large share of fraud directly.

---

## SQL Findings Summary

1. Overall fraud rate: 1.70% (highly imbalanced)
2. Jewelry has by far the highest category fraud rate (14.91%, ~9x average)
3. Foreign transactions: ~4x the fraud rate of domestic (5.62% vs 1.43%)
4. Late-night (1-4 AM) transactions: ~2.15x the fraud rate of rest of day, low volume
5. Fraud transactions average ~5.2x the amount of legitimate ones (₹1,014 vs ₹195)

---

## Part 2 — Z-Score Outlier Detection (Python)

**Method:** Computed z-score of `transaction_amount` ((value − mean) / std), flagged transactions above a threshold as suspicious, compared against actual `is_fraud` labels.

**At threshold z > 2:** 441 flagged, confusion matrix → TP=75, FP=366, FN=95, TN=9,464 → **Precision 17.0%, Recall 44.1%**

**Insight:** A single-variable (amount-only) z-score flag misses badly on precision — 83% of flagged transactions are false alarms. This confirms amount alone isn't enough; category, foreign-transaction, and time-of-day signals (found in SQL) would need to be combined for a usable system.

---

## Part 3 — Precision-Recall Trade-off

**Method:** Tested z-score thresholds from 1.5 to 3.5, computed precision/recall at each.

**Result:**
| Threshold | Precision | Recall | Flagged |
|---|---|---|---|
| 1.5 | 13.0% | 50.0% | 654 |
| 2.0 | 17.0% | 44.1% | 441 |
| 2.5 | 21.5% | 37.6% | 297 |
| 3.0 | 29.0% | 31.8% | 186 |
| 3.5 | 38.3% | 28.8% | 128 |

**Insight:** Classic precision-recall trade-off — stricter thresholds reduce false alarms but miss more fraud. No single threshold is "correct" without attaching a cost to each type of error, which Part 4 addresses.

---

## Part 4 — Cost-Benefit Threshold Analysis

**Assumptions:** False Negative cost = ₹1,014.40 (avg. fraud transaction amount lost) · False Positive cost = ₹50 (assumed manual review/verification cost)

**Result:**
| Threshold | FN | FP | Total Cost |
|---|---|---|---|
| 1.5 | 85 | 569 | ₹1,14,674 |
| 2.0 | 95 | 366 | ₹1,14,668 |
| 2.5 | 106 | 233 | ₹1,19,176 |
| 3.0 | 116 | 132 | ₹1,24,270 |
| 3.5 | 121 | 79 | ₹1,26,692 |

**Insight:** Threshold 2.0 minimizes total cost (essentially tied with 1.5), and cost rises steadily beyond that. Counter-intuitively, a looser threshold is cheaper overall — because the cost of a missed fraud (₹1,014) is ~20x the cost of a false alarm (₹50), so catching more fraud is worth tolerating more false positives. This is the core argument for basing thresholds on ₹ cost rather than on precision/recall alone.

---

## Final Monitoring Framework & Recommendation

**Recommended threshold:** z-score > 2.0 on transaction_amount as the base flag, combined with the SQL-derived risk signals — since amount alone only achieves 17% precision / 44% recall, layer in:
- **+1 risk point** for foreign transactions (4x baseline fraud rate)
- **+1 risk point** for Jewelry/Electronics/Travel category (highest category fraud rates)
- **+1 risk point** for late-night transactions (1-4 AM, 2.15x baseline rate)

**Escalation logic:** Transactions with z-score > 2 AND 2+ risk points → auto-hold for verification. Transactions with z-score > 2 but 0-1 risk points → soft flag for batch review (lower priority, reduces false-alarm friction on genuine high-value purchases).

**Review cadence:** Re-run threshold/cost analysis monthly as fraud patterns shift; track total cost (FN loss + FP review cost) as the core KPI, not precision/recall in isolation.
