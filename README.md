# 🛍️ H&M Recommendation System

**Neural Collaborative Filtering for Fashion Recommendations**

A machine learning-powered recommendation system built on real H&M transaction data, providing personalized fashion recommendations using PyTorch and neural collaborative filtering.

## 🎯 Project Overview

This system analyzes customer purchase patterns from H&M's dataset to generate personalized product recommendations. Built with modern deep learning techniques, it processes over 635k transactions from 372k users across 65k fashion items.

## ✨ Key Features

- **Neural Collaborative Filtering** - Advanced deep learning architecture
- **Real H&M Data** - Trained on authentic fashion transaction data
- **Personalized Recommendations** - Tailored suggestions for individual users
- **Interactive Demo** - Easy-to-use demonstration interface
- **Comprehensive Evaluation** - Multiple quality metrics and analysis
- **Scalable Architecture** - Handles large-scale fashion datasets

## 📊 System Performance

- **Users:** 372,056 unique customers
- **Products:** 64,889 fashion items
- **Transactions:** 635,766 purchase records
- **Model Size:** 55.9 MB
- **Recommendation Time:** ~30 seconds for top-10
- **Hit Rate@20:** 4.0%
- **Catalog Coverage:** 0.47%

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/kasia-dudek/hm-recommendation-system.git
cd hm-recommendation-system
pip install torch pandas scikit-learn numpy
```

### Data Preparation

1. Download H&M dataset from [Kaggle](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations)
2. Place files in `dane_hm/` directory:
   - `transactions_train.csv`
   - `articles.csv`
   - `customers.csv`

### Training the Model

```bash
# Train with 2% sample data (recommended for testing)
python simple_hm_recommender.py --action train --sample_size 0.02 --epochs 3

# Train with full dataset
python simple_hm_recommender.py --action train --sample_size 1.0 --epochs 5
```

### Generate Recommendations

```bash
# Get top-10 recommendations for a user
python simple_hm_recommender.py --action recommend --user_id "USER_ID" --top_k 10
```

### Interactive Demo

```bash
# Run demo with sample users
python demo.py
```

## 🏗️ Architecture

### Model Architecture
- **Embedding Dimensions:** 32 for users and items
- **Hidden Layers:** 16 neurons with ReLU activation
- **Output:** Sigmoid activation for rating prediction
- **Optimization:** Adam optimizer with early stopping
- **Regularization:** 0.1 dropout rate

### Project Structure

```
hm-recommendation-system/
├── simple_hm_recommender.py    # Main recommendation system
├── demo.py                     # Interactive demonstration
├── main.py                     # Extended interface with Azure
├── evaluate_hm_system_improved.py  # System evaluation
├── models/                     # Model definitions and training
├── data/                       # Data loading and preprocessing
├── utils/                      # Visualization utilities
├── azure/                      # Azure ML integration
└── dane_hm/                    # H&M dataset directory
```

## 📈 Evaluation Metrics

The system is evaluated using multiple metrics:

- **Precision@K** - Accuracy of top-K recommendations
- **Recall@K** - Coverage of relevant items in top-K
- **Hit Rate@K** - Percentage of users with at least one relevant item
- **F1-Score@K** - Harmonic mean of precision and recall
- **Catalog Coverage** - Diversity of recommended items
- **Popularity Bias** - Tendency to recommend popular items

### Sample Results

```
Hit Rate@20: 4.0%
Precision@20: 0.002
Recall@20: 0.04
Catalog Coverage: 0.47%
Popularity Bias: 0.0884
```

## 🎯 Sample Recommendations

**User 1:**
1. DONT USE ROLAND HOOD (Hoodie) - ID: 569974001
2. Zola dress (Dress) - ID: 890845007
3. Boyfriend (Sweater) - ID: 665508003

**User 2:**
1. Speedy Tee (T-shirt) - ID: 791587009
2. Superskinny (D) - ID: 810169016
3. Brittany LS (Top) - ID: 688558022

## 🔧 Configuration

### Training Parameters

```bash
--sample_size 0.02      # Fraction of data to use (0.01-1.0)
--epochs 3              # Number of training epochs
--batch_size 1024       # Training batch size
--learning_rate 0.001   # Adam learning rate
--embedding_dim 32      # Embedding dimensions
--hidden_dim 16         # Hidden layer size
--dropout 0.1           # Dropout rate
```

### Recommendation Parameters

```bash
--user_id "USER_ID"     # Target user for recommendations
--top_k 10              # Number of recommendations
```

## 🌟 Key Advantages

1. **Real Data** - Based on authentic H&M transactions
2. **Scalability** - Handles 372k users and 65k products
3. **Diversity** - Recommends across multiple fashion categories
4. **Performance** - Optimized with batch prediction and early stopping
5. **Usability** - Simple CLI interface and interactive demo
6. **Evaluation** - Comprehensive quality metrics

## 🔮 Future Enhancements

- **Content-Based Features** - Incorporate product metadata (color, size, brand)
- **Sequential Modeling** - Account for purchase order and timing
- **Contextual Recommendations** - Consider seasonality and fashion trends
- **Hybrid Approach** - Combine collaborative and content-based filtering
- **Real-Time Updates** - Dynamic model updates with new transactions

## 📚 Technical Details

### Dependencies

- **PyTorch** - Deep learning framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning utilities
- **NumPy** - Numerical computing

### Model Training

The system uses neural collaborative filtering with:
- User and item embeddings
- Multi-layer perceptron for interaction modeling
- Binary cross-entropy loss for implicit feedback
- Early stopping to prevent overfitting
