'''
    File name: data_cleaning.py
    Authors: Abhishek, Heemany, Nag
    Description: Performs data cleaning tasks such as
'''

# Importing dependencies
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sys

def print_horizontal_line():
    print ("-" * 40)

def data_normalization (df, num_cols):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    trans_mm = MinMaxScaler(feature_range = (-1., 1.))
    return trans_mm.fit_transform(df[num_cols])
    
def data_imputation(df, impute_strategy):
    from sklearn.preprocessing import Imputer
    imr = Imputer(missing_values=np.nan, strategy=impute_strategy, axis=0)
    imr = imr.fit(df)
    imputed_data = imr.transform(df.values)
    return pd.DataFrame(imputed_data, columns=list(df))
        
    
if __name__ == "__main__":
    # Reading the dataset
    print ("Reading the dataset")
    df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")
    print ("Dataset read")
    # print(df.describe())
    
    print_horizontal_line()
    
    print ("Normalizing numerical columns")
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df [num_cols] = data_normalization (df, num_cols)
    print ("Normalization complete")
    print_horizontal_line()
    
    
    print ("Handling categorical and ordinal data")
    df['Dependents'] = df['Dependents'].astype('category', ordered=True, categories=['0', '1', '2', '3+'])
    df['Gender'] = df['Gender'].astype('category')
    df['Married'] = df['Married'].astype('category')
    df['Education'] = df['Education'].astype('category', ordered=True,  categories=['Graduate', 'Not Graduate'])
    df['Self_Employed'] = df['Self_Employed'].astype('category')
    df['Property_Area'] = df['Property_Area'].astype('category')
    df['Loan_Status'] = df['Loan_Status'].astype('category')
    
    ordinal_cols = ['Dependents', 'Loan_Status', 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

    df[ordinal_cols] = df[ordinal_cols].apply(lambda x: x.cat.codes)
    
    print ("Imputing by mean")
    #taking a copy of loan_ID and deleting
    loan_id_df = df['Loan_ID']
    del df['Loan_ID']
    # Imputing by mean
    df = data_imputation(df, 'mean')
    # print (df.head(5))
    df.insert(0, 'Loan_ID', loan_id_df)
    df.to_csv('data/data_imputed_mean.csv')
    print ("Mean imputation complete")
    
    print ("Imputing by median")
    #taking a copy of loan_ID and deleting
    loan_id_df = df['Loan_ID']
    del df['Loan_ID']
    # Imputing by mean
    df = data_imputation(df, 'median')
    # print (df.head(5))
    df.insert(0, 'Loan_ID', loan_id_df)
    df.to_csv('data/data_imputed_median.csv')
    print ("Median imputation complete")
    sys.exit(0)