# DSC288R_Capstone-Project

## Project Description
User reviews on platforms such as Steam serve as essential indicators for game discovery; however, they vary considerably in sentiment, quality, and usefulness. This paper proposes a multi-modal machine learning framework that concurrently addresses three predictive tasks using the Steam Review Dataset (comprising over 100 million reviews): (1) recommendation classification—predicting whether a user recommends a game; (2) helpfulness prediction—assessing the usefulness of a review to other users; and (3) personalized game recommendation—ranking games that a user is likely to engage with next. We integrate review text features (TF-IDF and BERT embeddings) with behavioral signals (playtime, voting history, and engagement metrics) across gradient-boosted, collaborative filtering, and re-ranking models. LightGBM integrating both text and metadata achieves a ROC-AUC of 0.870 and increases the recall for the negative class from 50% to 66%. The BM25-weighted ALS method enhances Hit Rate@10 by 66% compared to a popularity baseline, whereas the Genre+Sentiment re-ranking approach attains the optimal balance between accuracy and diversity. Predicting helpfulness remains challenging (best R² = 0.061), indicating that platform-level factors beyond review content predominantly influence perceived usefulness.

## Repository Structure

```
DSC288R_Capstone-Project/
│
├── data/
│ ├── raw/ -> Initial raw CSV storage
| | └── (populated from the Kaggle downloads)
│ ├── intermediate/ -> Intermediate parquet storage
| | └── (populated from data ingest notebook)
│ └── final/ -> Final dataset storage
| | └── (populated from data ingest notebook)
│
├── models/
| └── (populated when notebooks are run)
│
├── notebooks/
│ ├── Clean_CF_ALS_Notebook.ipynb -> Alternating least squares experimentation and hypertuning
│ ├── Data_quality_EDA.ipynb -> Initial exploration of the dataset
│ ├── Diversity_Analysis_based_on_popularity_Sentiment.ipynb
│ ├── EDA_paige.ipynb
│ └── Review_helpfulness.ipynb
│ ├── advanced_modeling_paige.ipynb
│ └── target1_paige.ipynb
│ ├── Review_text_features_BERT_training.ipynb
│
├── visualizations/
│ ├── bert_complete_analysis.png
│ ├── genre_sentiment_comparison.png
│ └── performance_diversity_tradeoff.png
│
└── requirements.txt
└── README.md
```

Due to the large size of the dataset (>100M reviews), it cannot be hosted directly in this GitHub repository.
We considered using DVC for dataset management, but available hosting options required paid storage.
Therefore, the datasets must be downloaded manually from Kaggle.

## Workflow

To reproduce our results:

1. Download the datasets (see Data Access section).
2. Run `steam_ingestion.ipynb` to ingest and merge the datasets.
3. Run the **EDA notebooks** to reproduce exploratory analysis.
4. Run the **Recommendation Prediction notebooks**.
5. Run the **Helpfulness Prediction notebooks**.
6. Run `Review_text_features_BERT_training.ipynb` to train and evaluate recommender models.

## Data access
To reproduce this project:

1. Clone this repo
2. Download the datasets from Kaggle:

Steam Reviews Dataset
https://www.kaggle.com/datasets/kieranpoc/steam-reviews

Steam Games Metadata
https://www.kaggle.com/datasets/fronkongames/steam-games-dataset

3. Extract the datasets
4. Move the files:
5. all_reviews.csv -> data/raw/
6. games.csv -> data/raw/

7. Run the steam_ingestion.ipynb notebook sequentially. (This might take a while, we are creating parquets and merging them)

This will populate our processed and merged dataset into the final data folder. If this step is completed correctly there should be no need to change data paths in future notebooks.

## Initial EDA
To reproduce our results you will run the Data Quality and EDA notebook sequentially which will recreate our initial inspection of the dataset and the visualizations we used for exploratory analysis.  

## Recommendation Prediction models
1. Run target1_paige.ipynb sequentially for the baseline recommendation predictions
2. Run advanced_modeling_paige.ipynb sequentially for the advanced model recommendation predictions

## Review Helpfulness Prediction models:

This section contains the implementation for Task 2: Review Helpfulness Prediction. The main notebook is Review helpfulness.ipynb, which includes the preprocessing pipeline, feature engineering, and model training for predicting the helpfulness score of Steam reviews.
The experiments evaluate several regression models using both structured metadata and text features. Baseline models include Ridge Regression and Decision Tree Regression using structured features. More advanced models use Gradient Boosting Regression to capture nonlinear relationships. Text information from review content is incorporated using TF-IDF features and BERT embeddings, allowing the models to compare the impact of traditional bag-of-words representations versus contextual semantic features.
Running this notebook reproduces the experiments and results reported in the Helpfulness Prediction section of the final report.

1. Run review helpfulness.ipynb sequentially to recreate our results. 

## Recommender Models: 
1. Run clean_cf_als.ipynb to see experimentation and hyperparameter tuning on the ALS and ALS BM25 models. 
2. Run review_text_features_BERT training.ipynb to see BERT training as well as comparison to ALS and ALS BM25 from the same sample (this is what was used in the report and visualizations)
3. Run diversity_analysis_based on popular_sentiment.ipynb to see visualizations on the diversity of our recommender models

The experimentation that brought me to the final model for ALS and ALS with BM25 weighting is contained within the Clean CF ALS notebook. This experimentation was done on different samples from our final results and comparison which were completed within the models folder in the review_text_features_BERT training notebook. This is where the final analysis was done on the recommender models comparing ALS, ALS with BM25 weighting alongside the different models that implemented BERT embeddings in the recommendations. If you run through these notebooks you should be able to obtain the same results that we did throughout the course of our project. 
