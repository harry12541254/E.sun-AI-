# E.sun-AI-

本次信用卡詐欺偵測專案在資料上主要分成兩部分處理，分別為

##Overview
This project focuses on detecting credit card fraud. The data is processed in two main steps:

Preprocessing: Features are engineered to create additional attributes based on both static and dynamic credit card transaction data, such as transaction frequency and patterns.
Modeling: The model uses CatBoost, a gradient boosting algorithm, to handle categorical data and imbalance, providing superior performance in fraud detection.

```bash
E.sun-AI-/
├── Model/                 # Trained model files
├── Preprocess/            # Data preprocessing scripts
├── main.ipynb             # Main notebook for combining preprocessing and model
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Tech Stack
- **CatBoost** for classification
- **Python** for scripting and analysis

## Usage
- Run the `main.ipynb` to preprocess data, train the model, and generate predictions.
