# Uber Support Agent

An LLM-powered customer support pipeline that classifies customer issues, retrieves similar historical interactions, generates grounded responses, and identifies cases that may require human escalation.

## Problem Framing

For Uber support, a good agent should:

- identify the customer's primary issue correctly;
- draft a response supported by similar historical handling;
- provide a useful next step without inventing policies, refunds, or actions;
- identify cases that require human investigation.

### What I chose not to build

This project does not execute refunds, modify rides, access customer accounts,
or perform actions in Uber's internal systems. It only classifies the issue,
retrieves historical evidence, drafts a response, and recommends escalation
with a reason.

## Pipeline

```text
Customer Message
       |
       v
 Intent Classification
       |
       +------------------+
       |                  |
       v                  v
Similar-case         Escalation
 Retrieval            Decision
       |
       v
Grounded Response

## Dataset & Evaluation

This project uses the Twitter Customer Support (TWCS) dataset.

For Uber_Support, 56,193 usable customer-support pairs were extracted using tweet reply relationships.

## Evaluation sets
| Set                       | Size | Purpose                             |
| ------------------------- | ---: | ----------------------------------- |
| Development               |  100 | Taxonomy development and refinement |
| Golden                    |  200 | Held-out evaluation                 |
| Human response evaluation |   40 | Response-quality evaluation         |
| Human + LLM judge         |   20 | Judge-human agreement               |

The initial 14-intent taxonomy was refined to 11 intents after development-set labelling:
support_followup
fare_charge
driver_issue
uber_eats
lost_item
account_access_security
benefits_other
ride_booking_cancellation
location_app_issue
driver_metrics
refund
The golden set is imbalanced, so macro-F1 is reported alongside accuracy.

Intent Classification Results
| Model                        |  Accuracy |  Macro-F1 |
| ---------------------------- | --------: | --------: |
| Majority class               |     30.0% |      4.2% |
| TF-IDF + Logistic Regression |     45.5% |     29.6% |
| **LLM**                      | **56.0%** | **46.4%** |

The LLM outperformed both baselines on the held-out golden set, although performance varied considerably across intents, particularly for rare classes.

Escalation Results
| Metric    |    Result |
| --------- | --------: |
| Accuracy  | **69.5%** |
| Precision | **65.0%** |
| Recall    | **61.2%** |
| F1        | **63.0%** |

Confusion matrix:
|            | Predicted No | Predicted Yes |
| ---------- | -----------: | ------------: |
| Actual No  |           87 |            28 |
| Actual Yes |           33 |            52 |

Escalation was evaluated separately from intent because the customer's issue type and whether human intervention is required are distinct decisions.

Response Quality

Generated responses were manually evaluated on 40 examples using a 1–3 scale.
| Dimension       | Mean / 3 |
| --------------- | -------: |
| Helpfulness     | **2.40** |
| Grounding       | **2.62** |
| Actionability   | **2.40** |
| Appropriateness | **2.95** |

The system scored highest on appropriateness. Helpfulness and actionability were lower, reflecting a limitation of the historical support data: many retrieved responses are generic support-routing messages rather than substantive resolutions.

LLM-as-Judge Agreement

20 responses were evaluated by both a human and an LLM judge using the same 1–3 rubric.

Quadratic weighted Cohen's κ:
| Dimension       | Weighted κ | Exact Agreement |
| --------------- | ---------: | --------------: |
| Helpfulness     |     -0.118 |             30% |
| Grounding       |      0.000 |             40% |
| Actionability   |     -0.038 |             45% |
| Appropriateness |      0.000 |             90% |

The judge showed much stronger agreement with the human evaluator for appropriateness than for the other dimensions. The sample is small (n=20), so these results are treated as calibration findings rather than general estimates of judge reliability.

Top Failure Modes
1. benefits_other overprediction

@115877 Justice [link]

Human: support_followup
Model: benefits_other

The broad benefits_other class had only 0.12 precision, suggesting it is often used as a fallback for vague messages.

Hypothesis: insufficient evidence for a specific intent causes the model to fall back to the catch-all category.

2. Fare/charge vs. cancellation ambiguity

why is it that I get fined $5 if I cancel an Uber but get nothing in return...

Human: ride_booking_cancellation
Model: fare_charge

Hypothesis: the model focuses on the financial consequence instead of the underlying cancellation issue.

3. Driver issue vs. cancellation confusion

Cancellation is fine. He didn't even bother to notify me. I called him myself.

Human: ride_booking_cancellation
Model: driver_issue

Hypothesis: ride complaints frequently contain overlapping driver-behaviour and cancellation signals.

4. Rare-intent sparsity

Several intents had very little evaluation support:
| Intent               | Support |   F1 |
| -------------------- | ------: | ---: |
| `driver_metrics`     |       5 | 0.00 |
| `location_app_issue` |       2 | 0.18 |
| `refund`             |       3 | 0.44 |

Hypothesis: limited examples make it difficult to learn reliable boundaries for rare intents.

5. Multi-issue messages

What's with this $300 charge?... driver never even showed... ride I never took...

Human: fare_charge
Model: refund

Hypothesis: when several valid issues appear in one message, the model may select a secondary consequence rather than the primary complaint.

What Is Misleading About the Headline Number?

The LLM's 56.0% accuracy does not fully describe system quality.

The golden set is imbalanced, making macro-F1 important: the LLM achieved 46.4% macro-F1.

Intent accuracy also does not measure response usefulness or escalation quality.

Finally, historical support responses are frequently procedural. A retrieved case can therefore be highly similar to the customer's issue while still providing little information for actually resolving it.

Key Design Decisions
Selected Uber_Support because it provided a large set of varied support interactions.
Constructed customer-support pairs using tweet reply relationships.
Created a 100-example development set and an independent 200-example golden set.
Refined the taxonomy from 14 to 11 intents based on development labelling.
Separated intent classification from escalation.
Used majority-class and TF-IDF + Logistic Regression baselines.
Used balanced class weights for the Logistic Regression baseline.
Retrieved the top 3 similar historical cases.
Excluded golden customer messages from the retrieval corpus to prevent leakage.
Treated historical responses as evidence of prior handling rather than authoritative templates.
Added a primary-intent rule for multi-issue messages.
Used a 1–3 human rubric for response quality.
Used LLM-as-judge on 20 examples to measure judge-human agreement.
Evaluated response quality on 40 examples rather than relying only on automated judging.
Kept rare intents visible in evaluation rather than hiding their low performance through aggregation.
Reproduction
Requirements

Python 3.11+.

Install dependencies:

pip install -r requirements.txt
Dataset

Place the TWCS dataset at:

data/twcs.csv

The raw dataset is excluded from version control.

API

Create .env in the project root:

GROQ_API_KEY=your_api_key_here
Run

Open and run:

notebooks/01_explore_dataset.ipynb

The notebook contains the complete workflow:

Dataset processing
Uber pair extraction
Labelling
Baselines
LLM classification
Retrieval
Response generation
Escalation
Evaluation
Failure analysis

Saved evaluation outputs are included so headline results can be inspected without rerunning all LLM API calls.

Project Structure

SUPPORT-AGENT/
├── data/
│   ├── uber_dev_set.csv
│   ├── uber_golden_set.csv
│   ├── llm_golden_predictions.csv
│   ├── human_response_eval_40.csv
│   └── human_vs_llm_judge.csv
│
├── notebooks/
│   └── 01_explore_dataset.ipynb
│
├── src/
│   └── llm.py
│
├── .gitignore
├── requirements.txt
└── README.md

## Data Source

This project uses the **Customer Support on Twitter** dataset by Thought Vector.

Dataset:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/data

The dataset contains customer-support interactions between consumers and
brand support accounts on Twitter.

Limitations
Golden evaluation set contains only 200 examples.
Intent distribution is imbalanced.
Several intents have very small support.
Historical Twitter interactions may not represent current support workflows.
Historical responses are often generic support-routing messages.
LLM-as-judge agreement was evaluated on only 20 examples.
API-based evaluation depends on external model availability and rate limits.

Next Week
Collect more examples for rare intents.
Improve boundaries between fare, cancellation, driver, and refund intents.
Replace lexical retrieval with semantic embeddings.
Improve response specificity with structured support actions.
Expand human evaluation and recalibrate the LLM judge.
Evaluate the complete end-to-end pipeline with predicted rather than human-labelled intent passed to escalation.
