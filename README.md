# Sentiment Analysis Model

## Overview

This repository contains code for a sentiment analysis model using natural language processing techniques. The model is trained on a dataset of movie reviews to predict whether a review is positive or negative.

## Contents

- `sentiment_analysis.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `merged_train.csv`: CSV file containing the training dataset with movie reviews and sentiment labels.
- `merged_test.csv`: CSV file containing the testing dataset for evaluating the trained model.
- `README.md`: Documentation file providing an overview of the project.

## Dependencies

Ensure you have the following dependencies installed:

- Google Colab (for running the notebook in a Colab environment)
- Python 3
- Libraries: pandas, re, nltk, numpy, tensorflow, matplotlib, scikit-learn

Install the required libraries using the following command:

```bash
pip install pandas re nltk numpy tensorflow matplotlib scikit-learn
```

## Usage

1. Open and run the `sentiment_analysis.ipynb` notebook in a Jupyter environment or Google Colab.
2. The notebook includes sections for loading and preprocessing the dataset, building and training the sentiment analysis model, and evaluating the model on a test set.
3. Adjust hyperparameters, model architecture, or other settings as needed.
4. Monitor the training progress and evaluate the model's performance.

## Model Architecture

The sentiment analysis model is a Bidirectional LSTM neural network with an embedding layer. The model is trained to classify movie reviews into positive or negative sentiments.

## Evaluation

The model is evaluated on a separate test dataset. The key metrics, including loss and accuracy, are tracked over multiple training epochs. Additionally, a confusion matrix and accuracy score are calculated on the test set.

## Results

The model achieves a certain level of accuracy on the test set, as indicated by the evaluation metrics. However, further analysis, tuning, and potential adjustments are recommended for enhancing generalization performance.

## Predictions

After training, the model can be used to make predictions on new data. The notebook includes code for loading the trained model weights and making predictions on a test set.

## Author

[Your Name]

## License

This project is licensed under the [MIT License](LICENSE).
