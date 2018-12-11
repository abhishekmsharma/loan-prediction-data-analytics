from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

def get_train_test_split(df, y_col, test_split):
    Y = df[y_col]
    x_col = list(df)
    x_col.remove(y_col)
    print(x_col)
    X = df[x_col]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_split, random_state=2018)
    print("Train-Test split done.")
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)
    return X_train, X_test, y_train, y_test


def build_LR():
    clf = LogisticRegression()
    return clf


def build_SVC(kernel='rbf'):  # kerbel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}
    clf = SVC(kernel=kernel, probability=True, )
    return clf


def build_RFC(depth=None):
    clf = RandomForestClassifier(max_depth=depth)
    return clf


def build_NB():
    clf = GaussianNB()
    return clf

def get_prediction(model, x):
    return model.predict(x)


def model_evaluation(model, X_train, y_train, X_test, y_test):
    y_pred_train = get_prediction(model, X_train)
    y_pred_test = get_prediction(model, X_test)
    #     res_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
    #     print(res_df)
    acc_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
    rec_train = recall_score(y_true=y_train, y_pred=y_pred_train)
    prec_train = precision_score(y_true=y_train, y_pred=y_pred_train)

    print("Model evaluation of Training data: \n")
    print("Accuracy: ", acc_train)
    print("Recall: ", rec_train)
    print("Precision: ", prec_train)
    print()
    acc_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    rec_test = recall_score(y_true=y_test, y_pred=y_pred_test)
    prec_test = precision_score(y_true=y_test, y_pred=y_pred_test)

    print("Model evaluation of Test data: \n")
    print("Accuracy: ", acc_test)
    print("Recall: ", rec_test)
    print("Precision: ", prec_test)