# Intelligent Complaint Classification Using Localized Transformer Architectures

Automatic, natural-language classification and routing of e-commerce customer complaints — built as part of the AI Saturdays Lagos ML Cohort.

## Overview

E-commerce businesses receive complaints across websites, email, live chat, and mobile apps — many still triaged manually or by keyword matching. That breaks down fast: informal language, spelling errors, and multi-issue messages get misrouted, driving up response times and support workload.

This project trains a transformer-based classifier (BERT or a distilled variant) to sort incoming complaints into one of 10 categories, so they land with the right support team without manual review. Scope is deliberately limited to **classification and routing**, not complaint resolution — the system is a decision-support tool, and human oversight stays in the loop for uncertain or sensitive cases.

## Problem Statement

Manual and keyword-based triage fails to capture what a complaint is actually about, especially when the text is informal, misspelled, or covers more than one issue. Complaints about payments, orders, deliveries, refunds, accounts, and products end up in the wrong queue, which delays resolution and increases the load on support staff. This project addresses that by learning to classify complaints directly from their text.

## Categories

The model classifies each complaint into one of:

`billing` · `product_defect` · `delivery_shipping` · `refund_return` · `customer_service` · `account_access` · `fraud_unauthorized` · `warranty_repair` · `subscription_cancel` · `general_inquiry`

## Dataset

Built by combining existing public datasets rather than scraping or generating synthetic complaints:

| Source | Role |
|---|---|
| [Bitext Customer Support Training Dataset](data/) | Base intents, mapped into the 10-category taxonomy |
| CFPB consumer complaint data | Examples for `fraud_unauthorized` |
| Provided labeled complaint dataset | Additional training/eval coverage |

Data is standardized into a common text/label format and split with stratified sampling to preserve class balance. Class weighting is applied during training to reduce the effect of imbalance across categories.

**Known limitation:** consent/licensing terms for every source dataset aren't fully documented yet, and PII anonymization hasn't been finalized — both need to be resolved before any redistribution of the combined dataset. Consumers whose language, dialect, or channel is underrepresented in the source data are also more likely to be misclassified; see the Stakeholder Engagement section below for how this is being addressed.

## Evaluation

Performance is reported with precision, recall, F1, macro F1, and per-class F1 — with particular attention to minority classes, since overall accuracy alone would hide poor performance on smaller categories like `subscription_cancel` or `warranty_repair`.

> Results are not yet published in this README — training/evaluation is in progress. See `src/complaints-v2.ipynb` for the current run.

## Values Commitment

| Value | How it's operationalized |
|---|---|
| Fairness | Per-class evaluation, stratified splits, class weighting |
| Transparency | Precision/recall/F1 reported per class; dataset limitations documented |
| Accountability | Human oversight retained for uncertain, sensitive, or high-impact complaints |
| Accessibility | Free-text input instead of forced categories or keyword search |

## Stakeholder Engagement

Two stakeholder groups shaped the dataset and evaluation design:

- **E-commerce customers**, particularly those whose writing style or dialect is underrepresented in the source data — engaged through participatory design, co-writing adversarial/edge-case complaints (e.g. a `fraud_unauthorized` complaint in regional slang) that get folded back into training data.
- **Customer support personnel** — engaged through an ongoing user committee that reviews per-class metrics and maps weak spots (e.g. low recall on `billing`) to real workflow changes and re-allocated human review.

Full detail in `docs/Stakeholder Engagement Plan.pdf`.

## Repository Structure

```
├── data/
│   ├── Bitext_Sample_Customer_Support.csv
│   ├── cfpb_fraud_complaints-2026-08.csv
│   ├── train_complaints.csv
│   └── test_complaints.csv
├── docs/
│   ├── Data Card.pdf
│   ├── Impact statement.pdf
│   ├── Intelligent_Complaint_Classification_Project_Statement.pdf
│   └── Stakeholder Engagement Plan.pdf
├── src/
│   └── complaints-v2.ipynb
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt   # not yet in repo — add once dependencies are pinned
jupyter notebook src/complaints-v2.ipynb
```

## Limitations

- Model architecture (BERT vs. a distilled variant) is not finalized.
- No published accuracy/F1 numbers yet.
- Source-dataset licensing and PII handling need documentation before any data redistribution.
- Performance will vary across businesses and complaint styles not well covered by Bitext/CFPB data — this is being tracked, not assumed away.

## Author

**Dape Alexander Naanret** ([Anzaksen](https://anzaksen.github.io))
Data Scientist, Strategy and Results Delivery Office (SRDO), Plateau State Government

## Acknowledgment

Developed as part of the AI Saturdays Lagos Machine Learning Cohort.

## References

1. CFPB Consumer Complaint Database
2. Bitext Customer Support Training Dataset
