# config.py

# Database and table information
database = {
    'path': '../data/failure.db',
    'table': 'failure'
}

# File paths for additional data
file_paths = {
    'plot_folder': './plots'  # Path to the image folder
}

# Data processing configurations
preprocessing = {
    'numerical_imputer': 'mean',
    'scaler': 'StandardScaler'
}

# Parameter grids for classifiers
param_grid = {
    'RandomForest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'SVC': {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    },
    'DecisionTree': {
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    },
    'XGBoost': {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5, 7]
    }
}

# Targets to predict
targets = ['Failure A', 'Failure B', 'Failure C', 'Failure D', 'Failure E']

# Split configurations
split = {
    'test_size': 0.2,
    'random_state': 42
}

# Grid search configuration
grid_search = {
    'cv': 5,  # Cross-validation splits
    'n_iter': 6  # Number of iterations for RandomizedSearchCV
}
