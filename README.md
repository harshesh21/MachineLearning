Here is the updated, complete `README.md`. I have added a new section titled **"Adapting for a Different Market (Custom Training)"** that breaks down the exact pipeline steps for downloading new data, retraining the models, and generating the necessary PyTorch and XGBoost files.

You can copy and paste this directly over your current file.

***

# Real Estate AI Pricing Engine

## Overview
This project is an end-to-end Machine Learning pipeline that predicts real estate prices. Originally trained on the Houston, Texas market, it takes basic property specifications (Bedrooms, Bathrooms, Square Footage, and Zip Code) and processes them through an ensemble deep learning architecture to output an estimated market value.

The system is deployed as a REST API using FastAPI and includes a lightweight HTML/JS frontend for easy interaction. It is fully containerized using Docker for seamless cross-platform deployment.

## Tech Stack
* **Data Processing:** Pandas, Scikit-Learn
* **Machine Learning:** PyTorch (Deep Neural Network), XGBoost (Gradient Boosted Trees)
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend:** HTML, CSS, Vanilla JavaScript
* **Deployment:** Docker

## Architecture
1. **The Data:** Trained on the Kaggle USA Real Estate Dataset. Categorical variables (Zip Codes) are one-hot encoded, resulting in a sparse feature matrix.
2. **The Ensemble Model:**
   * **PyTorch Neural Network:** A 4-layer feedforward network (128 -> 64 -> 32 -> 1) trained with Huber Loss and Adam optimizer. Dropout layers (10%) were utilized to prevent overfitting.
   * **XGBoost:** A gradient-boosted tree model to capture non-linear geographical splits.
   * The final prediction is a 50/50 weighted average of both models.
3. **Data Imputation:** During inference, the API dynamically handles missing continuous variables by injecting median market values (e.g., defaulting to a 0.20-acre lot and a build year of 2005) to prevent zero-value penalties.

## Repository Structure
```text
├── main.py                    # FastAPI server and prediction logic
├── index.html                 # Frontend user interface
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker build instructions
├── utils/
│   └── download_kaggle_dataset.py # Script to pull fresh data via Kaggle API
├── train.py                   # Script to clean data and train the models
├── nn_weights.pth             # Trained PyTorch model weights (CPU-mapped)
├── xgb_model.json             # Trained XGBoost model
├── scaler.bin                 # Scikit-Learn standard scaler
└── features.bin               # List of the required training columns
```

## 🌍 Adapting for a Different Market (Custom Training)
This pipeline is modular. If you want to predict prices for a different city (e.g., Austin, TX, or Miami, FL), you can retrain the models from scratch using your own data.

**1. Download the Raw Data**
Use the included utility script to authenticate with the Kaggle API and download the latest USA Real Estate dataset.
```bash
python utils/download_kaggle_dataset.py
```

**2. Filter the Target Market**
Open the training script (`train.py`). Locate the Pandas filtering step and change the target city/state from `Houston` to your desired market. 

**3. Train the Models and Generate Artifacts**
Run the training pipeline. This will process the new CSV, scale the features, train the deep learning and XGBoost models, and automatically export the required production files.
```bash
python train.py
```
*(This will generate fresh `.pth`, `.json`, and `.bin` files tailored to your new market).*

**4. Update API Business Logic**
Open `main.py` and update the median injection defaults (e.g., `acre_lot` and `year_built`) to reflect the realistic averages of your new city. Update `index.html` to reflect a valid default zip code for the new area.

## Running the Application

There are two ways to run this project: using Docker (Recommended) or running it locally via Python.

### Option 1: Docker Deployment (Recommended)
Running the app via Docker ensures you do not run into Python versioning or dependency conflicts. 

1. Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.
2. Open your terminal in the project directory.
3. Build the container image:
   ```bash
   docker build -t real-estate-ml-engine .
   ```
4. Run the container:
   ```bash
   docker run -p 8000:8000 real-estate-ml-engine
   ```
5. Open `index.html` in any web browser to use the graphical interface, or navigate to `http://localhost:8000/docs` to use the interactive FastAPI Swagger UI.

### Option 2: Local Python Environment
If you prefer to run the server directly on your machine without Docker:

1. Ensure Python 3.10+ is installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The `requirements.txt` specifies the CPU-only version of PyTorch to save space).*
3. Start the FastAPI server:
   ```bash
   python -m uvicorn main:app --reload
   ```
4. Open `index.html` in your browser.

## API Usage

You can query the prediction endpoint programmatically.

**Endpoint:** `POST /predict`

**Request Body (JSON):**
```json
{
  "beds": 4,
  "baths": 2,
  "sqft": 2500,
  "zip_code": 77494
}
```

**Response:**
```json
{
  "beds": 4,
  "baths": 2,
  "sqft": 2500,
  "zip_code": 77494,
  "predicted_price": 389896.90
}
```
*(If a zip code is provided that was not present in the new training data, the API will return a warning flag and default to base market logic).*