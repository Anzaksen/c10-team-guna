# Intelligent Complaint Classification Using Localized Transformer Architectures

Automatic, natural-language classification and routing of e-commerce customer complaints — built as part of the AI Saturdays Lagos ML Cohort. 👋

## What This Project Does

E-commerce businesses handle a steady stream of complaints across websites, email, live chat, and mobile apps — and a lot of that still gets sorted manually or by keyword matching. That breaks down quickly once you factor in informal language, typos, and messages that mix two or three issues at once.

This project fine-tunes **ModernBERT** to read a complaint and route it straight to the right support category, cutting out the manual first pass. It's built as a decision-support tool rather than a replacement for support staff — human review stays in the loop for anything uncertain or sensitive.

## Problem Statement

Manual and keyword-based triage struggles to capture what a complaint is actually about, especially when the text is informal, misspelled, or covers more than one issue. Complaints about payments, orders, deliveries, refunds, accounts, and products end up in the wrong queue, which slows things down for both the customer and the support team. This project tackles that directly by learning to classify complaints from their raw text.

## Categories

The model sorts each complaint into one of ten categories:

`billing` · `product_defect` · `delivery_shipping` · `refund_return` · `customer_service` · `account_access` · `fraud_unauthorized` · `warranty_repair` · `subscription_cancel` · `general_inquiry`

## Dataset

Built by combining existing public datasets rather than scraping or generating synthetic complaints:

| Source | Role |
|---|---|
| Bitext Customer Support Training Dataset | Base intents, mapped into the 10-category taxonomy |
| CFPB consumer complaint data | Examples for `fraud_unauthorized` |
| Provided labeled complaint dataset | Additional training/eval coverage |

Data is standardized into a common text/label format, split with stratified sampling to preserve class balance, and class-weighted during training to account for the imbalance across categories. All sources have been reviewed and cleaned of personally identifiable information prior to training.

## Where Things Stand

Training and evaluation with ModernBERT are underway, tracked with precision, recall, F1, macro F1, and per-class F1 — with extra attention on smaller categories so overall accuracy doesn't mask weak spots. On held-out predictions submitted to Kaggle, the model currently scores an **F1 of 0.641**.

Full run in `src/complaints-v2.ipynb`.

## Values Commitment

| Value | How it's operationalized |
|---|---|
| Fairness | Per-class evaluation, stratified splits, class weighting |
| Transparency | Precision/recall/F1 reported per class |
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

Want to poke around or reproduce the results? Here's the quick path:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt   # not yet in repo — add once dependencies are pinned
jupyter notebook src/complaints-v2.ipynb
```

## Author

**Dape Alexander Naanret** ([Anzaksen](https://anzaksen.github.io))
Data Scientist.

## Acknowledgment

Built as part of the AI Saturdays Lagos Machine Learning Cohort — thanks to cohort mentors and peers for feedback on the taxonomy design and stakeholder engagement approach, and to the teams behind the Bitext and CFPB datasets for making this data publicly available.

## References

1. Consumer Financial Protection Bureau — [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
2. Bitext — [Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
3. Answer.AI / LightOn — [ModernBERT](https://huggingface.co/blog/modernbert)
