# Hate Speech Classification

This project implements hate speech classification using machine learning and deep learning approaches to classify tweets into three categories: hate speech, offensive language, or neither.

## Overview

The notebook compares two classification approaches:
1. **Baseline Model**: TF-IDF vectorization with Logistic Regression
2. **Deep Learning Model**: Fine-tuned DistilBERT transformer model

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn transformers datasets torch
```

Or install from the notebook by running the first cell.

## Dataset

The dataset (`data/labeled_data.csv`) contains labeled tweets with the following structure:
- **class**: Label (0 = hate speech, 1 = offensive language, 2 = neither)
- **tweet**: The text content of the tweet

## Setup Instructions

1. **Clone or download** this repository

2. **Ensure the dataset is in place**:
   - The dataset should be located at `data/labeled_data.csv`
   - If not present, download the dataset and place it in the `data/` directory

3. **Install dependencies** (if not already installed):
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn transformers datasets torch
   ```

4. **Download NLTK stopwords** (if needed):
   - The notebook will automatically download stopwords when run
   - Or manually run: `python -c "import nltk; nltk.download('stopwords')"`

## Running the Notebook

1. **Open the notebook**:
   ```bash
   jupyter notebook HateSpeechClassification.ipynb
   ```
   Or use JupyterLab, VS Code, or any other Jupyter-compatible environment.

2. **Run all cells**:
   - Use `Cell > Run All` from the menu, or
   - Press `Shift + Enter` to run cells sequentially

3. **Expected runtime**:
   - Baseline model: ~1-2 minutes
   - DistilBERT training: ~10-30 minutes (depending on hardware)
   - GPU acceleration (CUDA/MPS) will significantly speed up training

## Project Structure

```
HateSpeechClassification/
├── README.md                          # This file
├── HateSpeechClassification.ipynb    # Main notebook
└── data/
    ├── labeled_data.csv               # Dataset
    └── readme.md                      # Dataset information
```

## Notebook Sections

1. **Install dependencies and Import Libraries**: Setup and imports
2. **Load the dataset**: Data loading and exploration
3. **Preprocessing**: Text cleaning and normalization
4. **Train-test split + TF-IDF Baseline Model**: Baseline model implementation
5. **DistilBERT Fine-Tuning**: Deep learning model training and evaluation

## Hardware Requirements

- **Minimum**: CPU with 4GB RAM (will be slow for DistilBERT)
- **Recommended**: GPU (NVIDIA CUDA or Apple MPS) for faster training
- The notebook automatically detects and uses available GPU resources

## Results

The notebook evaluates both models using:
- Classification reports (precision, recall, F1-score)
- Confusion matrices
- Training loss curves (for DistilBERT)

## Notes

- The dataset is imbalanced (class 0 is underrepresented), so class weights are used during training
- Text preprocessing removes URLs, mentions, hashtags, numbers, and punctuation
- Stopwords are kept for transformer models but can be removed for the baseline
- Random seeds are set for reproducibility

## Troubleshooting

- **Tokenizers warning**: The multiprocessing warning is expected and can be ignored
- **Out of memory**: Reduce batch size in the DataLoader (currently 16)
- **Slow training**: Ensure GPU is being used (check device output in notebook)

## License

This project is for educational purposes.

