# Customer Segmentation Project

## Objective
Group customers into behavior-based segments using demographic and purchasing features. The goal is to help a business understand different customer groups and plan more relevant marketing strategies.

## Dataset
`data/customer_data.csv` contains 400 synthetic customer records. It includes:
- Customer ID
- Age
- Annual income (in thousands)
- Spending score (1–100)
- Monthly purchase frequency
- Average order value
- Customer tenure in months

The dataset is synthetic and created for learning/demo purposes. It does not contain real customer information.

## Methodology
1. Load and inspect the customer dataset.
2. Select demographic and behavioral features.
3. Standardize the features using `StandardScaler`.
4. Apply K-Means clustering with four clusters.
5. Profile the resulting segments.
6. Evaluate cluster separation with the silhouette score.
7. Visualize customer groups and segment sizes.

## Tools
Python, pandas, scikit-learn, matplotlib.

## Run the project
Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python customer_segmentation.py
```

## Output files
The script creates:
- `outputs/customer_segments.csv` — each customer with assigned segment
- `outputs/segment_profiles.csv` — average characteristics of each segment
- `outputs/model_evaluation.csv` — silhouette score
- `outputs/customer_segments_scatter.png` — income vs spending visualization
- `outputs/segment_customer_counts.png` — customers per segment chart

## Interpretation
Segment IDs are arbitrary labels, not rankings. Review `segment_profiles.csv` to understand the characteristics of each group before assigning business-friendly names or making marketing recommendations.

## Note
This is an independently prepared practice project using synthetic data. It is not based on Thiranex-provided data and does not imply approval by Thiranex. Review the outputs and follow the platform's submission rules before submitting.
