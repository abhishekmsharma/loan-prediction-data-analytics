import pandas as pd
import numpy as np
import os
import data_cleaning
import model_train_eval
import warnings
warnings.filterwarnings("ignore")


def read_file(filepath):
    df = pd.read_csv(filepath)
    df['Dependents'] = df['Dependents'].astype('category', ordered=True, categories=['0', '1', '2', '3+'])
    df['Gender'] = df['Gender'].astype('category')
    df['Married'] = df['Married'].astype('category')
    df['Education'] = df['Education'].astype('category', ordered=True, categories=['Graduate', 'Not Graduate'])
    df['Self_Employed'] = df['Self_Employed'].astype('category')
    df['Property_Area'] = df['Property_Area'].astype('category')
    df['Loan_Status'] = df['Loan_Status'].astype('category')
    return df


def prepare_data(data, num, cat, ord):
    data[num] = data_cleaning.data_normalization(data, num)
    data = data_cleaning.get_categorical(data, cat)
    data = data_cleaning.get_ordinal(data, ord)
    return data

def main():
    file = 'data/train_u6lujuX_CVtuZ9i.csv'
    # path = os.getcwd()
    # filepath = os.path.join(path, 'data')
    # print(filepath)
    # if (not os.path.isFile(file)) or (not os.path.exists(file)):
    #     print("file path not correct")
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    ordinal_cols = ['Dependents', 'Loan_Status']

    df = read_file(file)

    # Preparing subset data by dropping rows with missing values
    subset_df = df.dropna()
    subset_df = prepare_data(subset_df, num_cols, cat_cols, ordinal_cols)
    print(" Please close the plots to continue program execution. \n")
    data_cleaning.draw_distribution_plots(subset_df, num_cols)

    # Prepare train test split of 80/20
    X_train, X_test, y_train, y_test = model_train_eval.get_train_test_split(subset_df, 'Loan_Status', 0.2)

    lr = model_train_eval.build_LR()
    svc = model_train_eval.build_SVC('linear')
    rfc = model_train_eval.build_RFC()
    nb = model_train_eval.build_NB()

    print("Training and Evaluation resuts of Logistic Regression")
    lr.fit(X_train, y_train)
    model_train_eval.model_evaluation(lr, X_train, y_train, X_test, y_test)
    print()

    print("Training and Evaluation resuts of SVM classifier")
    svc.fit(X_train, y_train)
    model_train_eval.model_evaluation(svc, X_train, y_train, X_test, y_test)
    print()

    print("Training and Evaluation resuts of Random Forest classifier")
    rfc.fit(X_train, y_train)
    model_train_eval.model_evaluation(svc, X_train, y_train, X_test, y_test)
    print()

    print("Training and Evaluation resuts of Naive Bayes (Gaussian) classifier")
    nb.fit(X_train, y_train)
    model_train_eval.model_evaluation(svc, X_train, y_train, X_test, y_test)

    # Prepare same steps as above for imputed data
    # imputed_data = data_cleaning.data_imputation(df, 'mean')
    # print(imputed_data.shape)










if __name__=='__main__':
    main()