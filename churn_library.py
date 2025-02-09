"""
churn_library.py

This module contains functions for customer churn prediction modeling, including:
- Data ingestion 
- EDA (Exploratory Data Analysis)
- Feature engineering
- Model training (Random Forest and Logistic Regression)
- Model evaluation and result visualization

Date: February 2025
"""

# import libraries
import os
import shap
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info("SUCCESS: Data imported successfully from %s", pth)
        return df
    except FileNotFoundError as err:
        logging.error("ERROR: File not found at %s", pth)
        raise err
    except pd.errors.EmptyDataError as err:
        logging.error("ERROR: Empty CSV file at %s", pth)
        raise err


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:
        # Create images directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        
        # Churn distribution
        plt.figure(figsize=(20, 10))
        df['Churn'].hist()
        plt.title('Distribution of Churn')
        plt.savefig('./images/churn_distribution.png')
        plt.close()

        # Customer Age distribution
        plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        plt.title('Distribution of Customer Age')
        plt.savefig('./images/customer_age_distribution.png')
        plt.close()

        # Marital Status distribution
        plt.figure(figsize=(20, 10))
        df['Marital_Status'].value_counts('normalize').plot(kind='bar')
        plt.title('Distribution of Marital Status')
        plt.savefig('./images/marital_status_distribution.png')
        plt.close()

        # Total Transaction distribution
        plt.figure(figsize=(20, 10))
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        plt.title('Distribution of Total Transactions')
        plt.savefig('./images/total_transaction_distribution.png')
        plt.close()

        # Correlation matrix
        plt.figure(figsize=(20, 10))
        # 只选择数值类型的列进行相关性分析
        df_numeric = df.select_dtypes(include=['int64', 'float64'])
        sns.heatmap(df_numeric.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.title('Correlation Heatmap')
        plt.savefig('./images/correlation_heatmap.png')
        plt.close()

        logging.info("SUCCESS: EDA performed and images saved successfully")
    except Exception as err:
        logging.error("ERROR: EDA failed with error: %s", str(err))
        raise err

def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category
    '''
    try:
        df_copy = df.copy()
        for category in category_lst:
            category_groups = df_copy.groupby(category)[response].mean()
            df_copy[f'{category}_{response}'] = df_copy[category].map(category_groups)
        
        logging.info(f"SUCCESS: Encoded {len(category_lst)} categorical columns")
        return df_copy
        
    except Exception as err:
        logging.error(f"ERROR: Failed to encode categorical columns: {str(err)}")
        raise err

def perform_feature_engineering(df, response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        # 首先确保创建 Churn 列
        if 'Churn' not in df.columns:
            df['Churn'] = df['Attrition_Flag'].map(
                {'Existing Customer': 0, 'Attrited Customer': 1}
            )

        # 然后处理其他分类变量
        categorical_cols = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        
        # 对分类变量进行编码
        df = encoder_helper(df, categorical_cols, response)
        
        # 选择数值特征列
        keep_cols = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn'
        ]

        X = df[keep_cols]
        y = df[response]

        # 检查数据类型
        if not all(X[col].dtype.kind in 'biufc' for col in X.columns):
            raise ValueError("Non-numeric data found in features")

        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    except Exception as err:
        logging.error("ERROR: Feature engineering failed: %s", str(err))
        raise err

def classification_report_image(y_train,
                              y_test,
                              y_train_preds_lr,
                              y_train_preds_rf,
                              y_test_preds_lr,
                              y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    try:
        os.makedirs("images", exist_ok=True)

        plt.figure(figsize=(15, 5))
        plt.text(0.01, 1.25, 'Random Forest Train', fontsize=10)
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)), fontsize=10)
        plt.text(0.01, 0.6, 'Random Forest Test', fontsize=10)
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), fontsize=10)
        plt.axis('off')
        plt.savefig('./images/rf_results.png')
        plt.close()

        plt.figure(figsize=(15, 5))
        plt.text(0.01, 1.25, 'Logistic Regression Train', fontsize=10)
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), fontsize=10)
        plt.text(0.01, 0.6, 'Logistic Regression Test', fontsize=10)
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), fontsize=10)
        plt.axis('off')
        plt.savefig('./images/logistic_results.png')
        plt.close()

        logging.info("SUCCESS: Classification reports generated successfully")
    except Exception as err:
        logging.error("ERROR: Failed to generate classification reports: %s", str(err))
        raise err


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_pth), exist_ok=True)

        # Calculate feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(X_data.shape[1]), importances[indices])
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.tight_layout()
        plt.savefig(output_pth)
        plt.close()

        # Create SHAP plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        plt.figure(figsize=(20, 10))
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_pth.replace('.png', '_shap.png'))
        plt.close()

        logging.info(
            "SUCCESS: Feature importance plots saved to %s",
            output_pth)
    except Exception as err:
        logging.error(
            "ERROR: Failed to create feature importance plots: %s",
            str(err))
        raise err


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Initialize models
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        # Set parameters for grid search
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        # Perform grid search and train models
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        lrc.fit(X_train, y_train)

        # Make predictions
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        # Generate and save classification reports
        classification_report_image(
            y_train, y_test,
            y_train_preds_lr, y_train_preds_rf,
            y_test_preds_lr, y_test_preds_rf)

        # Plot ROC curve
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=ax, alpha=0.8)
        RocCurveDisplay.from_estimator(
            cv_rfc.best_estimator_,
            X_test,
            y_test,
            ax=ax,
            alpha=0.8)
        plt.savefig('./images/roc_curve.png')
        plt.close()

        # Generate feature importance plots
        feature_importance_plot(
            cv_rfc.best_estimator_,
            X_test,
            './images/feature_importances.png')

        # Save models
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')

        logging.info("SUCCESS: Models trained and saved successfully")
    except Exception as err:
        logging.error("ERROR: Failed to train models: %s", str(err))
        raise err


if __name__ == "__main__":
    # Load data
    data_df = import_data("./data/bank_data.csv")
    
    # Perform EDA
    perform_eda(data_df)
    
    # Perform feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(data_df)
    
    # Train and evaluate models
    train_models(X_train, X_test, y_train, y_test)