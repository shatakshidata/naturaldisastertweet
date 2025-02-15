#  Natural Language Processing with Disaster Tweets
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/nlp-getting-started)

##  Overview
This project is part of the **Kaggle NLP Competition - Natural Language Processing with Disaster Tweets**. The goal is to build a machine learning model that **predicts whether a given tweet is about a real disaster or not**.

Tweets can be **metaphorical, sarcastic, or unrelated to actual disasters**, making this a **challenging NLP task** that requires **contextual understanding beyond simple keyword matching**.

##  Dataset
The dataset consists of **tweets labeled as disaster (1) or non-disaster (0)** along with additional metadata like **keywords and locations**. 

 **Data Files**:
- `train.csv`: **7,613 tweets** with labels (`target = 1` for disaster, `target = 0` for not disaster).
- `test.csv`: **3,263 tweets** without labels (for predictions).
- `sample_submission.csv`: Format for Kaggle submission.

### ** Exploratory Data Analysis (EDA)**
Key insights from **EDA and feature analysis**:
- **Keyword Importance:** Some keywords (e.g., `"wildfire"`, `"earthquake"`, `"flood"`) are strongly correlated with real disasters.
- **Length of Tweets:** Disaster tweets tend to have **slightly more words and characters**.
- **Special Tokens Analysis:** Disaster tweets contain **more URLs (news links), hashtags, and exclamation marks**.
- **Location Relevance:** Many tweets **lack location data (~33% missing)**, making it less useful as a feature.

##  Approach
This project explores **multiple NLP approaches**, including:
1. **Baseline Models**: TF-IDF + Logistic Regression
2. **Deep Learning Approaches**: LSTMs, GRUs
3. **Transformer-based Models**:
   - **BERT** (`bert-base-uncased`)  
   - **DistilBERT** (`distilbert-base-uncased`)  
   - **Bertweet** (`vinai/bertweet-base`)  

##  Model Pipeline
1. **Text Preprocessing**:
   - Tokenization using **Hugging Face Transformers**
   - Lowercasing, removing URLs, mentions, hashtags
   - Handling missing keywords & locations (`no_keyword`, `no_location`)

2. **Feature Engineering**:
   - Word & character count features
   - Special token counts (URLs, hashtags, mentions, exclamations)
   - Keyword embeddings from pre-trained transformers

3. **Training & Validation Strategy**:
   - **3-Fold Stratified Cross-Validation**
   - **Early Stopping** to prevent overfitting
   - **Gradient Accumulation** for better memory efficiency

4. **Optimization Techniques**:
   - Learning Rate Scheduling (**Cosine Decay with Warmup**)
   - Mixed Precision Training (**FP16** for speedup)
   - Hyperparameter tuning using **Optuna/W&B Sweeps**

## Results
| **Model**        | **Validation Accuracy** | **F1 Score** | **Inference Time** |
|----------------|-------------------|------------|----------------|
| TF-IDF + Logistic Regression | 0.78 | 0.74 |  Very Fast |
| DistilBERT | 0.83 | 0.81 | Fast |
| BERTweet (Final Model) | **0.86** | **0.84** |  Moderate |

- The **BERTweet model achieved the highest accuracy & F1 score**, thanks to its training on social media text.
- **DistilBERT provided a great balance of speed & accuracy**.
- **Traditional models (TF-IDF + Logistic Regression) performed decently but lacked contextual understanding**.

## Future Improvements
Data Augmentation (Synonym Replacement, Back-Translation)
Stacking multiple Transformer models (BERTweet + DistilBERT)
Adversarial training to handle out-of-distribution tweets

## Contributors
Nikhita Shankar
