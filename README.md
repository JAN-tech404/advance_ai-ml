Advanced ML & AI Solutions Repository

This repository contains a collection of high-impact machine learning projects focusing on Natural Language Processing (NLP), Multimodal Deep Learning, and Automated Industrial Pipelines.
🛠️ Installation & Setup

To run these projects locally, ensure you have Python 3.8+ installed. You can install all necessary components using the following command:
 Bash

     pip install torch torchvision transformers datasets pandas numpy scikit-learn pillow matplotlib seaborn joblib

📂 Project Portfolio
1. BERT News Classification

    Objective: To develop a robust text classification system that accurately categorizes news articles into four distinct domains (World, Sports, Business, and Sci/Tech).

    Methodology/Approach:

        Architecture: Fine-tuned the bert-base-uncased transformer model.

        Data Pipeline: Utilized Hugging Face datasets for the AG News corpus and AutoTokenizer for text-to-tensor conversion.

        Training: Implemented the Trainer API with GPU acceleration support for efficient weight updates.

    Key Results:

        Achieved a Model Accuracy of 89.60%.

        Successfully exported the fine-tuned weights and tokenizer for offline deployment.

2. Multimodal Housing Price Prediction

    Objective: To create a regression model capable of predicting house prices by fusing structured tabular data (features) with unstructured visual data (images).

    Methodology/Approach:

        Synthetic Data Generation: Created a dataset of 1,000 houses, including custom-generated images using the PIL library based on house attributes.

        Dual-Stream Neural Network: * CNN Branch: A Convolutional Neural Network (PyTorch) to extract spatial features from house images.

            MLP Branch: A Multi-Layer Perceptron to process numerical features like square footage and age.

        Late Fusion: Concatenated the outputs of both branches into a final regression head.

    Key Results:

        Demonstrated successful CUDA-accelerated training for complex multimodal architectures.

        Generated comprehensive evaluation plots (Actual vs. Predicted) and error distribution analysis.

3. Telco Customer Churn Pipeline

    Objective: To build an automated, scalable machine learning pipeline that identifies customers likely to cancel their subscriptions.

    Methodology/Approach:

        Pipeline Engineering: Used Scikit-Learn Pipeline and ColumnTransformer to automate data cleaning, imputation, and feature scaling.

        Model Selection: Evaluated Logistic Regression and Random Forest models using 5-fold cross-validation.

        Hyperparameter Tuning: Applied GridSearchCV to optimize model performance on the Telco Churn dataset.

    Key Results:

        Developed a seamless "raw-data-to-prediction" workflow.

        Serialization: Saved the final best-performing estimator as telco_churn_pipeline.pkl for production use.

🗂️ Repository Organization
Plaintext

├── bert.ipynb                # NLP: Transformer-based news classification
├── housing_multimodal_ml.ipynb # CV & Regression: Image + Tabular fusion
├── ml_pipeline.ipynb         # Tabular: Automated end-to-end ML pipeline
├── output/                   # Directory containing model weights and plots
└── README.md                 # Project documentation
