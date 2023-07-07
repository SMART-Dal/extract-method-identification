from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_metrics_classicalml(true_labels,pred_labels):

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    return accuracy, precision, recall, f1

    
