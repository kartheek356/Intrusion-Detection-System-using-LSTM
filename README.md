🛡️📊 Intrusion Detection System using LSTM-RNN
Welcome! This project builds a high-performance Intrusion Detection System (IDS) for IoT environments using deep learning (LSTM-RNN). It processes cybersecurity attack data, detects threats in real time, and helps improve network defense systems.

🚀 Project Architecture
🔹 Data Preprocessing (Preprocessing.ipynb)
Loads the UNSW-NB15 dataset with 49 features and 9 attack types.

Performs:

Label encoding of categorical variables.

Feature scaling and normalization.

Train-test split for supervised learning.

SMOTE (Synthetic Minority Oversampling Technique) for class balancing.

🔹 IDS Model Training (LSTM-RNN-Model.ipynb)
Builds a deep learning model using:

4 LSTM layers

Dropout layers (rate: 0.75) to prevent overfitting

Dense output layer with softmax activation for multiclass classification.

Trained with categorical cross-entropy and Adam optimizer.

Achieves 98% accuracy in detecting network intrusions.

🛠 Tech Stack
Libraries & Tools:
Python (NumPy, Pandas, Scikit-Learn)

TensorFlow / Keras

Matplotlib / Seaborn

imbalanced-learn (SMOTE)

Dataset:
UNSW-NB15 dataset

🧠 Training Workflow
Data Cleaning
Remove irrelevant features and balance class distribution using SMOTE.

Model Construction
Sequential LSTM model trained on temporal network features.

Evaluation

Accuracy

Confusion matrix

Precision/Recall/F1 for each attack class

Optimization
Used RFE + Binary Grey Wolf Optimizer for feature selection (optional extension).

📁 Project Structure
Folder / File	Purpose
Preprocessing.ipynb	Data loading, cleaning, encoding, scaling, and SMOTE
LSTM-RNN-Model.ipynb	LSTM model architecture, training, and evaluation
README.md	Project documentation

✅ Results & Performance
Balanced detection of 9 attack types using LSTM time-sequence modeling.

Achieved 98% test accuracy, demonstrating strong generalization on unseen attack data.

Handles class imbalance and high-dimensional input via deep learning and feature selection.

⚙️ Quick Start
bash
Copy
Edit
# Set up environment
pip install -r requirements.txt  # (Add if you have one)

# Run preprocessing
Open Preprocessing.ipynb in Jupyter and run all cells

# Train model
Open LSTM-RNN-Model.ipynb and execute the training workflow
