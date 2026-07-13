# Spam Email Classifier

Train and use a simple spam email classifier based on email text content.

This project implements a dependency-free Multinomial Naive Bayes classifier using bag-of-words token counts. It ships with a tiny sample CSV so the workflow runs immediately, and you can replace it with a larger dataset that has `label` and `text` columns.

## Train

```powershell
python train.py
```

The training command writes:

- `outputs/spam_model.pkl`: trained model
- `outputs/evaluation.txt`: accuracy, precision, recall, and F1 score

## Predict

```powershell
python predict.py "Claim your free prize now"
```

## Use Your Own Dataset

Create a CSV with these columns:

```csv
label,text
spam,"Win money now"
ham,"Can we meet tomorrow?"
```

Then train with:

```powershell
python train.py --data path\to\emails.csv
```

For larger production datasets, a scikit-learn pipeline with `TfidfVectorizer` and either `MultinomialNB` or a linear SVM is a natural next step.
