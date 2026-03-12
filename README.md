# DSC288R_Capstone-Project

##Project Description
User reviews on platforms such as Steam serve as essential indicators for game discovery; however, they vary considerably in sentiment, quality, and usefulness. This paper proposes a multi-modal machine learning framework that concurrently addresses three predictive tasks using the Steam Review Dataset (comprising over 100 million reviews): (1) recommendation classification—predicting whether a user recommends a game; (2) helpfulness prediction—assessing the usefulness of a review to other users; and (3) personalized game recommendation—ranking games that a user is likely to engage with next. We integrate review text features (TF-IDF and BERT embeddings) with behavioral signals (playtime, voting history, and engagement metrics) across gradient-boosted, collaborative filtering, and re-ranking models. LightGBM integrating both text and metadata achieves a ROC-AUC of 0.870 and increases the recall for the negative class from 50% to 66%. The BM25-weighted ALS method enhances Hit Rate@10 by 66% compared to a popularity baseline, whereas the Genre+Sentiment re-ranking approach attains the optimal balance between accuracy and diversity. Predicting helpfulness remains challenging (best R² = 0.061), indicating that platform-level factors beyond review content predominantly influence perceived usefulness.

## Repository Structure

```
DSC288R_Capstone-Project/
│
├── data/
│ ├── raw/ # Raw dataset ingestion
│ ├── intermediate/ # Intermediate parquet files
│ └── final/ # Final merged dataset
│
├── models/
│ ├── Diversity_Analysis_based_on_popularity_Sentiment.ipynb
│ ├── Review_text_features_BERT_training.ipynb
│ ├── advanced_modeling_paige.ipynb
│ └── target1_paige.ipynb
│
├── notebooks/
│ ├── Clean_CF_ALS_Notebook.ipynb
│ ├── Data_quality_EDA.ipynb
│ ├── Diversity_Analysis_based_on_popularity_ver1.ipynb
│ ├── EDA_paige.ipynb
│ └── Review_helpfulness.ipynb
│
├── visualizations/
│ ├── bert_complete_analysis.png
│ ├── genre_sentiment_comparison.png
│ └── performance_diversity_tradeoff.png
│
└── README.md
```

Due to the large size of the dataset (>100M reviews), it cannot be hosted directly in this GitHub repository.
We considered using DVC for dataset management, but available hosting options required paid storage.
Therefore, the datasets must be downloaded manually from Kaggle.

## Workflow

To reproduce our results:

Download the datasets (see Data Access section).
Run `steam_ingestion.ipynb` to ingest and merge the datasets.
Run the **EDA notebooks** to reproduce exploratory analysis.
Run the **Recommendation Prediction notebooks**.
Run the **Helpfulness Prediction notebooks**.
Run `Review_text_features_BERT_training.ipynb` to train and evaluate recommender models.
