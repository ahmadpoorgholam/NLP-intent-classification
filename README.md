# Intent Classification Model

This project implements a text classification model to predict intents from input text. It uses various machine learning techniques to process and classify Turkish text data.

## Setup

1. Install required libraries:
   ```
   pip install pandas numpy scipy matplotlib seaborn scikit-learn
   ```

2. Ensure you have the following files in your project directory:
   - `intent_data.csv`: CSV file containing the text data and corresponding intents
   - `turkce-stop-words`: File containing Turkish stop words

## Usage

Run the Jupyter notebook or Python script to:

1. Load and preprocess the data
2. Visualize intent distribution
3. Train and compare different classifiers
4. Perform hyperparameter tuning
5. Evaluate the final model

## Key Components

- Data loading and preprocessing
- Text vectorization using TF-IDF
- Model comparison (Random Forest, Linear SVC, Naive Bayes, Logistic Regression)
- Hyperparameter tuning using Grid Search
- Model evaluation using classification report and confusion matrix

## Results

The Linear SVC model achieved the best performance with an accuracy of approximately 64% on the test set. Detailed performance metrics for each intent class are provided in the classification report.

## Future Improvements

- Experiment with more advanced text preprocessing techniques
- Try deep learning models like LSTM or BERT
- Collect more training data to improve model performance
