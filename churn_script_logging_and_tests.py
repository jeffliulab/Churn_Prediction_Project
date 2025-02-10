"""
This module contains test functions for churn_library.py 
and implements logging functionality.

Author: Pang Liu
Date: February 2025
"""

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    
    input:
            import_data: function to be tested
    output:
            None
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    
    input:
            perform_eda: function to be tested
    output:
            None
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        perform_eda(df)
        
        # Check if all expected images are created
        assert os.path.exists("./images/churn_distribution.png")
        assert os.path.exists("./images/customer_age_distribution.png")
        assert os.path.exists("./images/marital_status_distribution.png")
        assert os.path.exists("./images/total_transaction_distribution.png")
        assert os.path.exists("./images/correlation_heatmap.png")
        
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: One or more expected images not found")
        raise err
    except Exception as err:
        logging.error("Testing perform_eda: An error occurred: %s", str(err))
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    
    input:
            encoder_helper: function to be tested
    output:
            None
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        
        df_encoded = encoder_helper(df, category_lst)
        
        # Check if new columns were created
        for category in category_lst:
            assert f'{category}_Churn' in df_encoded.columns
        
        # Check if the new columns contain numeric values
        for category in category_lst:
            assert df_encoded[f'{category}_Churn'].dtype in ['float64', 'int64']
        
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Encoded columns not created correctly")
        raise err
    except Exception as err:
        logging.error("Testing encoder_helper: An error occurred: %s", str(err))
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    
    input:
            perform_feature_engineering: function to be tested
    output:
            None
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        
        # Check if the splits are not empty
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        # Check if the splits have the expected proportions (70-30 split)
        assert len(X_train) > len(X_test)
        assert abs(len(X_train) - 0.7 * len(df)) < 10  # allowing small deviation
        
        # Check if X has all expected features
        expected_features = [
            'Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
            'Income_Category_Churn', 'Card_Category_Churn'
        ]
        assert all(feature in X_train.columns for feature in expected_features)
        
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Features not created correctly")
        raise err
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: An error occurred: %s",
            str(err))
        raise err


def test_train_models(train_models):
    '''
    test train_models
    
    input:
            train_models: function to be tested
    output:
            None
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df)
        train_models(X_train, X_test, y_train, y_test)
        
        # Check if models are saved
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
        
        # Check if model result images are created
        assert os.path.exists("./images/rf_results.png")
        assert os.path.exists("./images/logistic_results.png")
        assert os.path.exists("./images/roc_curve.png")
        assert os.path.exists("./images/feature_importances.png")
        assert os.path.exists("./images/feature_importances_shap.png")
        
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Models or result images not created correctly")
        raise err
    except Exception as err:
        logging.error("Testing train_models: An error occurred: %s", str(err))
        raise err


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./images", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    # Run all tests
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)