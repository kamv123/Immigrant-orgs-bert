# Identifying Immigrant-Serving Nonprofits Using BERT

> NLP · Applied ML
> 
> A BERT-based pipline to identify immigrant-serving nonprofit organizations at scale using IRS data.

## Project Overview

This project builds a BERT-based text classification pipeline to identify immigrant- and migrant-serving nonprofit organizations at scale from the IRS Business Master File (BMF). Because immigrant-serving organizations are not explicitly labeled in federal nonprofit data, this project uses supervised learning with curated positive examples to infer organizational focus from text-based identifiers.

## Pipeline Overview
Raw nonprofit data is cleaned and standardized, embedded using pretrained BERT models, and classified to identify immigrant-serving organizations at scale.

The final output is a cleaned, deduplicated list of predicted immigrant-serving nonprofits, suitable for research, policy analysis, or nonprofit sector mapping.

## Goals

Researchers and practitioners often lack a reliable way to identify immigrant-serving nonprofits in large administrative datasets. Existing classifications are incomplete or inconsistent, and manual review does not scale.

This project addresses that gap by:
- Leveraging pretrained BERT embeddings to capture semantic meaning in organization names
- Combining multiple trusted data sources to construct a labeled training set
- Applying the trained model to the full IRS BMF to generate new predictions

## Data Sources

IRS Business Master File (BMF) – universe of U.S. nonprofit organizations

Curated positive-label datasets, including:

GuideStar immigration- and migrant-related nonprofit lists

Manually compiled known immigrant-serving organizations

All data is processed for consistency (name standardization, EIN cleaning, deduplication).

## Methodology
1. Label Construction

Positive class (1): Known immigrant- or migrant-serving nonprofits from curated sources

Negative class (0): Randomly sampled nonprofits from the BMF, excluding close name matches to positive examples using similarity thresholds

2. Text Embeddings

Organization names (and relevant text fields) are encoded using pretrained BERT / Sentence-Transformer models

Embeddings capture semantic similarity beyond keyword matching

3. Model Training

A logistic regression classifier is trained on BERT embeddings

Model performance is evaluated and probability thresholds are tuned to balance recall and precision

4. Inference at Scale

The trained classifier is applied to the full IRS BMF

Predictions are cleaned, deduplicated, and exported as final outputs

## Representative Code Snippets

Below are selected excerpts from the private implementation that illustrate key components of the pipeline.

### Organization Name Cleaning
```python
import pandas as pd
import re

def clean_org_name(name):
    if pd.isnull(name):
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(
        r'\b(the|inc|llc|foundation|co|corp|association|assn|org|of|and|a|an)\b',
        '',
        name
    )
    return re.sub(r'\s+', ' ', name).strip()
```
### EIN Standardization for Cross-Dataset Matching
```python
BMF_sub['EIN_clean'] = (
    BMF_sub['EIN']
    .astype(str)
    .str.replace(r'\D', '', regex=True)
    .str.zfill(9)
)

df_final['EIN_clean'] = (
    df_final['EIN']
    .astype(str)
    .str.replace(r'\D', '', regex=True)
    .str.zfill(9)
)
```
### Similarity-Based Negative Sampling
```python
threshold = 90

matched_to_exclude = match_df.loc[
    match_df['similarity_score'] >= threshold,
    'matched_bmf_name'
].unique()

BMF_non_immigrant = BMF_sub[
    ~BMF_sub['org_clean'].isin(matched_to_exclude)
]
```
### Baseline Text Classifier (TF-IDF + Logistic Regression)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000
)

X = vectorizer.fit_transform(combined_df['org_clean'])
y = combined_df['label']

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
```
### BERT Model Initialization
```python
import torch
from transformers import BertTokenizer, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()
```

## Key Takeaways

Semantic embeddings significantly improve identification of mission-driven nonprofits compared to keyword rules

Administrative data can be meaningfully extended using modern NLP methods

Careful label construction and negative sampling are critical for applied NLP in policy contexts

## Potential Extensions

Fine-tuning BERT on nonprofit mission statements.

Expanding classification to other issue areas (housing, legal aid, refugee services).

Incorporating geospatial or financial features from IRS filings.

> **Code Access:**  
> The full implementation is currently private while the project is under review.  
> Code access can be provided upon request.
