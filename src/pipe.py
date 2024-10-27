import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import recall_score, make_scorer, confusion_matrix, precision_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score
import config
import re
import os

class ClassifierPipeline:
    def __init__(self, config):
        self.config = config
        self.classifiers = self._initialize_classifiers()
        self.best_models = {}
        self._create_plot_folder()

    def _create_plot_folder(self):
        # Create plot_folder if not exists
        os.makedirs(self.config.file_paths['plot_folder'], exist_ok=True)

    def _initialize_classifiers(self):
        return {
            'RandomForest': (RandomForestClassifier(class_weight='balanced', random_state=self.config.split['random_state']), self.config.param_grid['RandomForest']),
            'SVC': (SVC(class_weight='balanced', probability=True, random_state=self.config.split['random_state']), self.config.param_grid['SVC']),
            'KNN': (KNeighborsClassifier(), self.config.param_grid['KNN']),
            'DecisionTree': (DecisionTreeClassifier(class_weight='balanced', random_state=self.config.split['random_state']), self.config.param_grid['DecisionTree']),
            'GradientBoosting': (GradientBoostingClassifier(random_state=self.config.split['random_state']), self.config.param_grid['GradientBoosting']),
            'XGBoost': (XGBClassifier(eval_metric='logloss', random_state=self.config.split['random_state']), self.config.param_grid['XGBoost']),
        }

    def return_temp1(self, length):
        # Extract decimal from string
        if '°F' in length:
            temp = re.search(r'\d+\.\d+', length)
            if temp is not None:
                fahrenheit_temp = float(temp.group(0))
                return (fahrenheit_temp - 32) * 5.0 / 9.0
        else:    
            temp = re.search(r'\d+\.\d+', length)
            if temp is not None:
                return float(temp.group(0))

    def load_data(self):
        conn = sqlite3.connect(self.config.database['path'])
        query = f"SELECT * FROM {self.config.database['table']}"
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def clean_data(self, df):
        # Make a copy of df
        df = df.copy()

        # Separating composite values
        if 'Model' in df.columns:
            df[['Model_Name', 'Year']] = df['Model'].str.split(',', expand=True)

        if 'Factory' in df.columns:
            df[['City', 'Country']] = df['Factory'].str.split(',', expand=True)

        # Converting data type
        df["Temperature_C"] = df.apply(lambda row: self.return_temp1(row['Temperature']), axis=1)

        # Converting sign to positive
        df['RPM'] = df['RPM'].abs()

        # Removing duplicate rows
        df.drop_duplicates(inplace=True)

        # Removing outliers
        df = df[df['Temperature_C'] <= 150]

        # Dropping redundant columns
        redundant_cols = ['Car ID', 'Temperature', 'Model', 'Factory']
        df.drop(columns=redundant_cols, inplace=True, errors='ignore')

        return df

    def create_pipeline(self):
        """Create a preprocessing pipeline for numerical and categorical features.

        This method constructs a column transformer to handle numerical and 
        categorical features separately, applying appropriate scaling and encoding.
        """

        # Load the dataset
        df = self.load_data()

        # Clean the data
        df = self.clean_data(df)

        self.feature_cols = df[[col for col in df.columns if col not in self.config.targets]]

        # Separate numerical data from categorical data
        numerical_features = self.feature_cols.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.feature_cols.select_dtypes(include=['object']).columns

        scaler = StandardScaler() if self.config.preprocessing['scaler'] == 'StandardScaler' else MinMaxScaler()

        # Combining both numerical and categorical pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=self.config.preprocessing['numerical_imputer'])), 
                    ('scaler', scaler)
                ]), numerical_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), 
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )
        return preprocessor

    def run_random_search(self):
        """Perform random search over multiple classifiers for each target.

        This method loads and cleans the dataset, separates predictors and targets, 
        and then iterates through classifiers to find the best model for each target 
        using RandomizedSearchCV.
        """

        # Load the dataset
        df = self.load_data()

        # Clean the data
        df = self.clean_data(df)

        self.feature_cols = df[[col for col in df.columns if col not in self.config.targets]]

        # Seperate predictor variables from target variables
        X = self.feature_cols
        y = df[self.config.targets]

        # Create pipeline
        preprocessor = self.create_pipeline()
        recall_scorer = make_scorer(recall_score, average='binary')

        # Initialise empty lists for targets and models for storing scores
        all_cv_scores = {target: [] for target in self.config.targets}
        model_names = {target: [] for target in self.config.targets}

        for target in self.config.targets:
            # Create hold-out set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y[target], test_size=self.config.split['test_size'], random_state=self.config.split['random_state']
            )

            best_recall = 0
            best_model = None
            evaluation_metrics = {}
            all_scores = {name: [] for name in self.classifiers.keys()}  # Collect scores for each model

            for name, (model, param_dist) in self.classifiers.items():
                # Create pipeline
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

                # Run RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=param_dist,
                    n_iter=self.config.grid_search['n_iter'],
                    scoring=recall_scorer,
                    cv=self.config.grid_search['cv'],
                    n_jobs=-1,
                    random_state=self.config.split['random_state']
                )

                random_search.fit(X_train, y_train)

                # Store best estimator
                best_model_from_search = random_search.best_estimator_
                y_pred = best_model_from_search.predict(X_test)
                y_prob = best_model_from_search.predict_proba(X_test)[:, 1]

                # Evaluate best model from random search
                recall = recall_score(y_test, y_pred, average='binary')
                precision = precision_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')
                roc_auc = roc_auc_score(y_test, y_prob)

                print(f'Best recall for {name} on {target} with RandomizedSearchCV: {recall}')
                print(f'Precision: {precision}, F1 Score: {f1}, ROC AUC: {roc_auc}')

                # Store evaluation metrics
                evaluation_metrics[name] = {
                    'recall': recall,
                    'precision': precision,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'y_test': y_test,  # Store y_test for precision-recall curve
                    'y_pred': y_pred   # Store y_pred for precision-recall curve
                }

                # Update model with the highest recall
                if recall > best_recall:
                    best_recall = recall
                    best_model = random_search.best_estimator_

                # Evaluate the best model using cross-validation
                cv_scores = cross_val_score(best_model_from_search, X, y[target], cv=self.config.grid_search['cv'], scoring='recall')
                all_cv_scores[target].append(cv_scores)  # Store scores for the specific target
                model_names[target].append(name)  # Store model name for the specific target

            # Store best model for each target
            self.best_models[target] = (best_model, best_recall)

            # Print confusion matrix for the best model
            self.print_metrics_and_confusion_matrix(best_model, X_test, y_test, target)

            # Visualize evaluation metrics
            self.visualize_metrics(evaluation_metrics, target)

        # Create individual box plots for cross-validation scores for each target
        for target in self.config.targets:
            self.plot_cv_scores(all_cv_scores[target], model_names[target], target)

    def print_metrics_and_confusion_matrix(self, model, X_test, y_test, target):
        """Print the confusion matrix, precision, recall, F1 score, and ROC AUC score for the best model."""
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for the positive class
        cm = confusion_matrix(y_test, y_pred)

        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        print(f'Confusion Matrix for Best Model on {target}:')
        print(cm)
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print(f'ROC AUC Score: {roc_auc:.2f}\n')



    def plot_cv_scores(self, cv_scores, model_names, target):
        """Create a box plot of cross-validation scores for each model."""
        # Create a DataFrame from scores
        scores_df = pd.DataFrame(cv_scores, index=model_names).T

        # Create the box plot
        plt.figure(figsize=(12, 6))
        scores_df.boxplot()
        plt.title(f'Cross-Validation Scores for Each Model on Target {target}')
        plt.xlabel('Models')
        plt.ylabel('Recall Scores')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.config.file_paths['plot_folder'], f'cross_validation_scores_box_plot_{target}.png'))
        plt.close()

    def visualize_metrics(self, metrics, target):
        """Visualize evaluation metrics and save plots."""
        metrics_df = pd.DataFrame(metrics).T

        # Precision-Recall Curve
        plt.figure(figsize=(10, 6))
        for name, values in metrics.items():
            precision, recall, _ = precision_recall_curve(values['y_test'], values['y_pred'])
            plt.plot(recall, precision, label=f'{name} (AP = {average_precision_score(values["y_test"], values["y_pred"]):.2f})')

        plt.title(f'Precision-Recall Curve for Each Model on {target}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig(os.path.join(self.config.file_paths['plot_folder'], f'precision_recall_curve_{target}.png'))
        plt.close()

        # ROC AUC Curve
        plt.figure(figsize=(10, 6))
        for name, values in metrics.items():
            fpr, tpr, _ = roc_curve(values['y_test'], values['y_pred'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(values["y_test"], values["y_pred"]):.2f})')

        plt.title(f'ROC Curve for Each Model on {target}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(os.path.join(self.config.file_paths['plot_folder'], f'roc_curve_{target}.png'))
        plt.close()





