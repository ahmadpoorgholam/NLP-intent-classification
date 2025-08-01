# Intent Classification Model

This project implements a text classification model to predict intents from input text. It now includes a standalone deep learning script built with PyTorch.

## Setup

1. Install required libraries:
   ```
   pip install torch
   ```

2. Ensure you have the following file in your project directory:
   - `intent_data.csv`: CSV file containing the text data and corresponding intents

## Usage

To train and evaluate the neural network classifier run:

```
python intent_classifier.py
```

The script reads the dataset, trains a small neural network, and prints the model's accuracy along with the label mapping used for training.

## Key Components

- Data loading and preprocessing
- Simple PyTorch model using averaged word embeddings
- Train/test split and accuracy evaluation

## Results

Accuracy will vary depending on random initialization and the small dataset size.

## Future Improvements

- Experiment with more advanced text preprocessing techniques
- Try deeper architectures like LSTMs or BERT
- Collect more training data to improve model performance
