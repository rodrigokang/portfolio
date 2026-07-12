# Dataset

This project uses the **Amazon Reviews 2023** public dataset to analyse customer feedback through Natural Language Processing (NLP). The analyses focus on customer-written product reviews from the **All_Beauty** product category.

The dataset was published by the **McAuley Lab** at the University of California, San Diego, and represents one of the largest publicly available collections of Amazon customer reviews for academic research.

# Data Source

**Dataset**

Official project site: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)

**Reference**

Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024).

*Bridging Language and Items for Retrieval and Recommendation.*

arXiv.

The official project website provides:

- complete dataset documentation;
- download links for all review and metadata categories;
- licensing information;
- citation instructions;
- links to the accompanying research paper.

# Downloaded Files

The project uses the following files from the **All_Beauty** category.

| File | Description |
|------|-------------|
| `All_Beauty.jsonl` | Customer reviews, ratings, timestamps, and auxiliary review information. |
| `meta_All_Beauty.jsonl` | Product metadata including titles, descriptions, categories, pricing, and images. |

Only `All_Beauty.jsonl` is required for the current implementation.

The metadata file is included to support future extensions involving product-level analyses.

# Directory Structure

The repository separates raw, intermediate, and processed datasets.

```text
data/
│
├── raw/
│   ├── All_Beauty.jsonl
│   └── meta_All_Beauty.jsonl
│
├── interim/
│
└── processed/
```

The training notebook expects the original dataset under:

```text
data/raw/
```

Processed datasets generated during execution are stored in:

```text
data/processed/
```

# Dataset Validation

After loading the sampled dataset, a basic validation step is performed.

Representative statistics from the current implementation are:

| Metric | Value |
|----------------------|------:|
| Sampled reviews | 50,000 |
| Variables | 10 |
| Duplicate rows | 45 |
| Missing values | 0 |

The duplicated records represent less than 0.1% of the sampled dataset and are removed during preprocessing.

# Data Dictionary

| Variable | Description |
|----------|-------------|
| `rating` | Customer rating (1–5 stars). |
| `title` | Review title. |
| `text` | Review text. |
| `images` | Images attached to the review. |
| `asin` | Amazon Standard Identification Number (ASIN). |
| `parent_asin` | Parent product identifier grouping product variants. |
| `user_id` | Anonymous customer identifier. |
| `timestamp` | Review submission date and time. |
| `helpful_vote` | Number of helpful votes received. |
| `verified_purchase` | Indicates whether Amazon verified the purchase. |

# Target Definition

Customer ratings are converted into three sentiment classes.

| Rating | Sentiment |
|---------|-----------|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

This mapping creates a balanced three-class sentiment classification problem suitable for supervised machine learning.

# Sampling Strategy

The complete **All_Beauty** dataset contains several hundred thousand reviews.

To improve reproducibility and reduce execution time during exploratory analysis, the notebook generates a reusable random sample containing **50,000 reviews** using **reservoir sampling**.

The sampling procedure is:

- deterministic through a fixed random seed;
- independent of file ordering;
- completed in a single pass through the source file;
- statistically unbiased;
- reusable across notebook executions.

Once generated, the sampled dataset is stored as:

```text
data/processed/all-beauty-sample.jsonl
```

Subsequent executions automatically reuse the sampled file.

# Train/Test Split

To reduce information leakage between products, the project uses a **product-grouped train/test split**.

Reviews belonging to the same product are never distributed across both training and testing datasets.

This evaluation strategy provides a more realistic estimate of model performance when predicting sentiment for previously unseen products.

# Notes

The notebook intentionally does not download the dataset automatically.

Keeping the original files outside the notebook ensures:

- transparent data provenance;
- reproducibility;
- compatibility with future dataset updates;
- independence from external download services.

Although the metadata file is not used during model training, it remains available for future analyses involving product attributes, brands, pricing, or category-level comparisons.