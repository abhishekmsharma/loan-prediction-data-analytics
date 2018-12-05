# This is just a standalone module for multiclass roc plot.
import time
import numpy as np
from sklearn.svm import SVC
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
from matplotlib import pyplot as plt

def plot_ROC(model, c0_X_tst, c1_X_tst, c2_X_tst, c3_X_tst, c4_X_tst, y0_act
                , y1_act, y2_act, y3_act, y4_act, issvm=0):
    n_classes = 10
    if issvm:
        y0_score = model.decision_function(c0_X_tst)
        y1_score = model.decision_function(c1_X_tst)
        y2_score = model.decision_function(c2_X_tst)
        y3_score = model.decision_function(c3_X_tst)
        y4_score = model.decision_function(c4_X_tst)

    else:
        y0_score = model.predict_proba(c0_X_tst)
        y1_score = model.predict_proba(c1_X_tst)
        y2_score = model.predict_proba(c2_X_tst)
        y3_score = model.predict_proba(c3_X_tst)
        y4_score = model.predict_proba(c4_X_tst)

    fpr0, fpr1, fpr2, fpr3, fpr4 = dict(), dict(), dict(), dict(), dict()
    tpr0, tpr1, tpr2, tpr3, tpr4 = dict(), dict(), dict(), dict(), dict()
    roc_auc0, roc_auc1, roc_auc2, roc_auc3, roc_auc4, = dict(), dict(), dict(), dict(), dict()
    for i in range(n_classes):
        fpr0[i], tpr0[i], _ = roc_curve(y0_act[:, i], y0_score[:, i])
        roc_auc0[i] = auc(fpr0[i], tpr0[i])
        fpr1[i], tpr1[i], _ = roc_curve(y1_act[:, i], y1_score[:, i])
        roc_auc1[i] = auc(fpr1[i], tpr1[i])
        fpr2[i], tpr2[i], _ = roc_curve(y2_act[:, i], y2_score[:, i])
        roc_auc2[i] = auc(fpr2[i], tpr2[i])
        fpr3[i], tpr3[i], _ = roc_curve(y3_act[:, i], y3_score[:, i])
        roc_auc3[i] = auc(fpr3[i], tpr3[i])
        fpr4[i], tpr4[i], _ = roc_curve(y4_act[:, i], y4_score[:, i])
        roc_auc4[i] = auc(fpr4[i], tpr4[i])

    # Then interpolate all ROC curves at this points
    # f, axes = plt.subplots(2, 3, figsize=(20, 10))
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'peru', 'g', 'orchid', 'y', 'brown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr0[i], tpr0[i], color=color, lw=2,
                              label='ROC curve of digit {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc0[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for 1st Char')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('plots/svm_1.png', bbox_inches='tight')

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr1[i], tpr1[i], color=color, lw=2,
                              label='ROC curve of digit {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc1[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for 2nd Char')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('plots/svm_2.png', bbox_inches='tight')

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr2[i], tpr2[i], color=color, lw=2,
                              label='ROC curve of digit {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc2[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for 3rd Char')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('plots/svm_3.png', bbox_inches='tight')

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr3[i], tpr3[i], color=color, lw=2,
                              label='ROC curve of digit {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc3[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for 4th Char')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('plots/svm_4.png', bbox_inches='tight')

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr4[i], tpr4[i], color=color, lw=2,
                              label='ROC curve of digit {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc4[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for 5th Char')
    plt.legend(loc="lower right")
    plt.show()
plt.savefig('plots/svm_5.png', bbox_inches='tight')
