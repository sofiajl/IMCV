###################################################
## Evaluation metrics
###################################################


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pred_threshold(predictions, thr):
    predictions = np.asarray([1. if val > thr else 0. for val in predictions])
    
    return predictions

def plt_cm_metrics(y_test, y_pred):

    '''
        Class getting the confusion matrix metrics and plotting it

        - Input:
            - y_test, np.array, target of real images 
            - y_pred, np.array, target prediction
    '''

    print('Acuraccy: %.3f' % accuracy_score(y_test, y_pred))
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    fig, ax = plt.subplots(figsize=(5,5))
    cmp.plot(ax=ax)


def calc_metric(y_test, y_pred, metric):

    '''
        Class calculate either accuracy, precision or recall

        - Input:
            - y_test, np.array, target of real images 
            - y_pred, np.array, target prediction
            - metric, string, metric to be calculated
    '''

    if metric == 'acc':
        return accuracy_score(y_test, y_pred)
        
    elif metric == 'precision':
        return precision_score(y_test, y_pred)

    elif metric == 'recall':
        return recall_score(y_test, y_pred)


def plt_metrics_thr(y_test, y_pred):

    '''
        Class for plotting the curve of accuracy, precision and recall for different range of thresholds
        
        - Input:
            - y_test, np.array, target of real images 
            - y_pred, np.array, target prediction
    '''

    threshold = np.arange(0.0, 1.0, 0.05)
    acc_thres = []
    precision_thres = []
    recall_thres = []
    
    for i in threshold:
        y_thrs_pred = pred_threshold(y_pred, thr = i)
        acc_thres.append(calc_metric(y_test, y_thrs_pred, 'acc'))
        precision_thres.append(calc_metric(y_test, y_thrs_pred, 'precision'))
        recall_thres.append(calc_metric(y_test, y_thrs_pred, 'recall'))

    acc = np.array(acc_thres)
    precision = np.array(precision_thres)
    recall = np.array(recall_thres)

    fig, ax = plt.subplots(1,3, figsize=(15, 4))
    
    # Plotting the Graph
    ax[0].plot(threshold, acc)
    ax[0].set_title("Accuracy by threshold")
    ax[0].set_xlabel("Threshold")
    ax[0].set_ylabel("Accuracy")
    
    ax[1].plot(threshold, precision)
    ax[1].set_title("Precision by threshold")
    ax[1].set_xlabel("Threshold")
    ax[1].set_ylabel("Precision")
    
    ax[2].plot(threshold, recall)
    ax[2].set_title("Recall by threshold")
    ax[2].set_xlabel("Threshold")
    ax[2].set_ylabel("Recall")



def plot_far_frr(y_test, y_pred):
    
    '''
        Class for plotting the curve of FAR and FRR for different range of thresholds
    
        - Input:
            - y_test, np.array, target of real images 
            - y_pred, np.array, target prediction
    '''
    
    threshold = np.arange(0.0, 1.0, 0.05)
    far_thres = []
    frr_thres = []
    
    for i in threshold:
        y_thrs_pred = pred_threshold(y_pred, thr = i)
        cm = confusion_matrix(y_test, y_thrs_pred)

        TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
        # False Acceptance Rate
        far_thres.append(FP/(FP + TN))
        
        # False Rejection Rate
        frr_thres.append(FN/(FN + TP))

    FAR = np.array(far_thres)
    FRR = np.array(frr_thres)

    plt.plot(threshold, FAR, label = 'FAR')
    plt.plot(threshold, FRR, label= 'FRR')
    plt.legend(loc="upper left")
    plt.xlabel("Threshold")
    plt.show()
        



def all_metrics(y_test, y_pred):

    
    '''
        Class for calculatin accuracy, precision recall, FAR and FRR for different range of thresholds

        - Input:
            - y_test, np.array, target of real images 
            - y_pred, np.array, target prediction
    '''

    threshold = np.arange(0.0, 1.0, 0.1)
    acc_thres = []
    precision_thres = []
    recall_thres = []
    far_thres = []
    frr_thres = []
    
    for i in threshold:
        y_thrs_pred = pred_threshold(y_pred, thr = i)
        acc_thres.append(calc_metric(y_test, y_thrs_pred, 'acc'))
        precision_thres.append(calc_metric(y_test, y_thrs_pred, 'precision'))
        recall_thres.append(calc_metric(y_test, y_thrs_pred, 'recall'))

        cm = confusion_matrix(y_test, y_thrs_pred)
        TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
        # False Acceptance Rate
        far_thres.append(FP/(FP + TN))
        
        # False Rejection Rate
        frr_thres.append(FN/(FN + TP))

    acc = np.array(acc_thres)
    precision = np.array(precision_thres)
    recall = np.array(recall_thres)
    FAR = np.array(far_thres)
    FRR = np.array(frr_thres)


    return threshold, acc, precision, recall, FAR, FRR