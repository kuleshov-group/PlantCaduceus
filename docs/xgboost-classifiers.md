# XGBoost Classifiers

[← Back to main README](../README.md)

## Training XGBoost classifiers

Train custom classifiers on top of PlantCAD embeddings for specific annotation tasks (e.g., TIS, TTS, splice sites).

**Purpose**: Fine-tune prediction performance for specific annotation tasks using supervised learning.

**Data format**: Training data should follow the format used in our [cross-species annotation dataset](https://huggingface.co/datasets/kuleshov-group/cross-species-single-nucleotide-annotation/tree/main/TIS).

```bash
python src/train_XGBoost.py \
    -train train.tsv \
    -valid valid.tsv \
    -test test_rice.tsv \
    -model 'kuleshov-group/PlantCaduceus_l20' \
    -output ./output \
    -device 'cuda:0'
```

**Expected outputs:**

- Trained XGBoost classifier (`.json` file)
- Performance metrics on validation/test sets
- Feature importance analysis

## Using pre-trained XGBoost classifiers

We provide pre-trained XGBoost classifiers for common annotation tasks in the [`classifiers`](../classifiers) directory.

**Available classifiers:**

- TIS (Translation Initiation Sites)
- TTS (Translation Termination Sites)
- Splice donor/acceptor sites

```bash
python src/predict_XGBoost.py \
    -test test_rice.tsv \
    -model 'kuleshov-group/PlantCaduceus_l20' \
    -classifier classifiers/PlantCaduceus_l20/TIS_XGBoost.json \
    -device 'cuda:0' \
    -output ./output
```

**Expected output**: Predictions with confidence scores for each sequence in your test data.
