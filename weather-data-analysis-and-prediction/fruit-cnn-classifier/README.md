# Fruit CNN Classifier

A small, self-contained fruit image classifier that trains a convolutional neural network on a generated dataset of apples, bananas, and oranges.

The project uses only:

- Python 3
- NumPy
- Pillow

It avoids TensorFlow/PyTorch so it can run in lightweight environments while still demonstrating the CNN pipeline end to end.

## Project Structure

```text
fruit-cnn-classifier/
  train.py              # Generate data, train CNN, evaluate, save outputs
  predict.py            # Load saved model and predict one image
  requirements.txt      # Minimal package list
  artifacts/            # Created after training
```

## Run Training

```powershell
python train.py
```

Training creates:

- `artifacts/dataset/` with generated fruit images
- `artifacts/fruit_cnn_model.npz` with trained weights
- `artifacts/sample_predictions.png` with example predictions

## Predict One Image

After training:

```powershell
python predict.py artifacts/dataset/test/apple/apple_000.png
```

## Notes

This is a compact learning project. The images are synthetic so the model can be trained quickly without downloading a dataset. For real-world performance, replace `generate_dataset()` with a real image folder such as:

```text
dataset/
  train/
    apple/
    banana/
    orange/
  test/
    apple/
    banana/
    orange/
```

Then keep the same preprocessing, model, and evaluation flow.
