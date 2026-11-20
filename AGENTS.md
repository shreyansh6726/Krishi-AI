# Krishi AI Agent Instructions

## Project overview

Krishi AI is a Windows-based agriculture project with three local models and a Groq-backed web chatbot:

- Crop recommendation: scikit-learn Random Forest.
- Crop disease detection: TensorFlow CNN.
- Crop price prediction: CatBoost regressor.
- Conversational interface: Groq API with a Python standard-library web server.

## Machine-learning model context

The repository contains three locally trained models. Training scripts read
datasets from the local KaggleHub cache at
`C:\Users\shrey\.cache\kagglehub\datasets`; they do not download data
automatically. Each trainer selects the most recently modified cached dataset
version and saves its artifact under that model's `model` directory.

### Crop recommendation

- **Purpose:** Recommend a crop from soil nutrients and weather conditions.
- **Model:** scikit-learn `RandomForestClassifier` with 100 trees and
  `random_state=42`.
- **Training dataset:** KaggleHub
  `atharvaingle\crop-recommendation-dataset`. The CSV uses `N`, `P`, `K`,
  `temperature`, `humidity`, `ph`, and `rainfall` as features and `label` as
  the crop target. The trainer uses an 80/20 `train_test_split` with
  `random_state=42`.
- **Artifact:** `crop_recommendation\model\crop_recommendation_model.pkl`.
- **Usage:** `native_language_nlp\groq.py` loads the artifact with `joblib`,
  converts the seven required values to numeric columns with the exact
  training feature names, and returns the predicted crop. The chatbot first
  asks for any missing values rather than inventing them.

### Crop disease detection

- **Purpose:** Classify a plant-leaf image into a disease or healthy-plant
  class and report the top prediction with confidence.
- **Model:** TensorFlow/Keras convolutional neural network built from four
  convolution/max-pooling blocks, a 512-unit dense layer, 0.5 dropout, and a
  softmax output whose class count is read from the dataset. Images are
  resized to 128x128 RGB pixels and normalized from `[0, 255]` to `[0, 1]`.
  Training uses batches of 32 for 10 epochs with Adam at learning rate
  `0.001`.
- **Training dataset:** KaggleHub
  `vipoooool\new-plant-diseases-dataset`. The trainer locates its `train` and
  `valid` directories and uses the folder names as labels.
- **Artifact:** `crop_disease_detection\model\crop_disease_cnn_model.keras`.
- **Usage:** `native_language_nlp\groq.py` loads the Keras model, applies the
  same 128x128 resize and normalization to an uploaded image, finds the
  highest-probability class, and loads labels from the cached dataset's
  `train` directory. The chatbot requires an attached image before running
  this model.

### Crop price prediction

- **Purpose:** Predict an agricultural market's modal price.
- **Model:** CatBoost `CatBoostRegressor` with 1,000 iterations, learning rate
  `0.05`, depth 6, categorical-feature handling, MAE evaluation, and
  `random_seed=42`. The trainer uses an 80/20 random split and early stopping
  after 50 rounds without improvement on the held-out set.
- **Training dataset:** KaggleHub
  `arjunyadav99\indian-agricultural-mandi-prices-20232025`. The target is
  `Modal_Price`. Categorical features are `STATE`, `District Name`, `Market
  Name`, `Commodity`, `Variety`, and `Grade`. Numeric features are
  `Min_Price`, `Max_Price`, and year/month/day/day-of-week values derived from
  `Price Date`.
- **Artifact:** `crop_price_prediction\model\crop_price_prediction_model.cbm`.
- **Usage:** `native_language_nlp\groq.py` loads the CatBoost artifact and
  builds one row using the same six categorical and six numeric fields as
  training, then returns the predicted modal price. The chatbot requests all
  required fields before prediction.

The Groq-backed chatbot classifies each message as `recommendation`, `price`,
`disease`, or `general`. Only the first three intents call local models;
general questions remain routed to Groq. Local model results must not be
claimed unless the corresponding model actually loads and returns a result.

## Repository conventions

- Use Windows paths in commands and documentation.
- Use the existing `venv` environment:
  ```powershell
  .\venv\Scripts\python.exe
  ```
- Do not install packages globally.
- Keep API keys and credentials in the root `.env` file.
- Never commit `.env`, model secrets, downloaded datasets, or credentials.
- Preserve the existing model file locations under each project’s `model` directory.
- Keep frontend assets under `web`.
- Keep the web-server entry point at the repository root as `server.py`.

## Dataset conventions

Training scripts must use the local KaggleHub cache:

```text
C:\Users\shrey\.cache\kagglehub\datasets
```

Do not reintroduce automatic dataset downloads when changing the trainers. Fail with a clear error when a required cached dataset is missing.

## Chatbot conventions

- Keep the Groq API key server-side.
- Load configuration from `.env`.
- Route crop recommendation, price prediction, and disease detection to the local models.
- Do not fabricate missing model inputs; ask the user for the required values.
- Keep user messages as plain text in the browser.
- Escape assistant content before rendering Markdown or HTML.
- Do not claim that a local model ran unless it actually returned a result.

## Code changes

- Inspect related scripts and model paths before editing.
- Prefer small, focused changes.
- Reuse existing feature names and preprocessing exactly during inference.
- Surface missing files, invalid inputs, and dependency errors explicitly.
- Avoid broad exception handlers that hide failures.
- Do not modify unrelated working-tree changes.
- Do not commit changes unless explicitly requested.

## Validation

For Python changes, run focused syntax checks:

```powershell
.\venv\Scripts\python.exe -m py_compile path\to\changed_file.py
```

For JavaScript changes, run:

```powershell
node --check path\to\changed_file.js
```

For chatbot changes, verify that:

1. `server.py` starts from the repository root.
2. `http://127.0.0.1:8000` serves the web client.
3. `/api/chat` returns a clear error when `GROQ_API_KEY` is missing.
4. Local model files are found at their documented paths.

Do not run full model training unless it is necessary to validate the requested change.
