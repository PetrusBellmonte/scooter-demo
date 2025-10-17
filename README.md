# Scooter case study
This project is a quick 4-5 hour exploration of a provided dataset and an attempt to create a decent predictor, to perform a case study for an application process (and to test streamlit). The goal is to perform prediction of scooter usage on unseen data. This repo should be taken as is. While it was fun to play around, I do not indeed on developing it further ;)

The data is not included as I do not own the rights to it!

## Usage
### Streamlit-Website
Install everything needed from the pyproject.toml and run
```
streamlit run Overview.pyi
```
### Scooter Model CLI Usage

The CLI allows you to train and evaluate the scooter demand prediction model from the command line.

#### Basic Usage

```sh
python model.py
```
This will train the model on `scooter.csv` and print R² scores for both train and test splits (default test split is 0.2). The data file needs to be in the directory.

#### Options

- `--train <path>`: Path to the training dataset CSV (default: `scooter.csv`)
- `--eval <path>`: Optional path to a separate evaluation dataset CSV. If provided, the model is trained on `--train` and evaluated on `--eval`.
- `--test_split <float>`: Ratio of data to use for test split if no evaluation set is provided (default: 0.2)

#### Example: Train and Evaluate

```sh
python model.py --train scooter.csv --test_split 0.3
```

#### Example: Train and Evaluate on Separate Dataset

```sh
python model.py --train scooter.csv --eval new_data.csv
```

#### Output

The CLI prints R² scores for each target on the train and test/eval sets.

## Acknowledgement
AI was heavily used in writing this code. This is neither especially good nor complex code. It is only intended to create a predictor fast while at the same time getting submittable results!