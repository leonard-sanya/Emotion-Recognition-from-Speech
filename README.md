# Emotion-Recognition-from-Speech

This repository contains a project focused on emotion recognition from speech using the Toronto Emotional Speech Set (TESS) dataset, leveraging pre-existing libraries and tools for machine learning and audio processing.

# Key Features

Dataset: TESS (Toronto Emotional Speech Set)
2800 audio recordings
Two actresses, aged 26 and 64
Seven emotional states: anger, disgust, fear, happiness, pleasant surprise, sadness, neutral
Libraries and Tools Used

# Librosa:
# Key functions:
- librosa.load(): Load audio files
- librosa.feature.mfcc(): Extract Mel-Frequency Cepstral Coefficients (MFCCs)
- librosa.feature.zero_crossing_rate(): Calculate zero-crossing rate
- librosa.feature.spectral_rolloff(): Compute spectral roll-off
# Scikit-learn:
# Key components:
- sklearn.model_selection.train_test_split(): Split data into training and testing sets
- sklearn.preprocessing.StandardScaler(): Standardize features
- sklearn.svm.SVC(): Support Vector Machine classifier
- sklearn.ensemble.RandomForestClassifier(): Random Forest classifier
- sklearn.metrics.classification_report(): Evaluate classification performance
# TensorFlow/Keras:
# Key features:
- keras.models.Sequential(): Build sequential neural networks
- keras.layers.Dense(): Fully connected layers
- keras.optimizers.Adam(): Optimizer for training
- keras.callbacks.EarlyStopping(): Stop training when a monitored metric has stopped improving
# Project Workflow

# Data Preprocessing:

- Load audio files using Librosa
- Extract features (MFCCs, zero-crossing rate, spectral roll-off)
- Feature Extraction:

- Compute relevant audio features for each recording
# Model Training:

- Split data into training and testing sets
- Standardize features using Scikit-learn
- Train machine learning models (SVM, Random Forest)
- Train neural networks using TensorFlow/Keras
# Model Evaluation:

- Evaluate model performance using classification reports and accuracy metrics
