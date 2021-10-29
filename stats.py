
def per_class_counts(y_true, y_pred):
    threshold = 0.5
    class_count = len(y_true[0])
    tp = [0]*class_count
    tn = [0]*class_count
    fp = [0]*class_count
    fn = [0]*class_count
    for i in range(len(y_pred)):
        probs = y_pred[i]
        trues = y_true[i]
        for j in range(len(probs)):
            # per-class tp, fp, fn and tn?
            if probs[j] >= threshold:
                if trues[j] == 0:
                    #false positive
                    fp[j] += 1
                else:
                    #true positive
                    tp[j] += 1
            elif probs[j] < threshold:
                if trues[j] == 1:
                    #false negative
                    fn[j] += 1
                else:
                    #true negative
                    tn[j] += 1

    return tp, tn, fp, fn


def safe_per_element_division(numerator, denominator):
    results = [0] * len(denominator)
    for i in range(len(denominator)):
        if denominator[i] == 0:
            results[i] = -1
        else:
            results[i] = numerator[i]/denominator[i]
    return results


def per_class_f1_score(tp, fp, fn):
    total_false = [x + y for (x, y) in zip(fp, fn)]
    denominator = [x + 0.5 * y for (x, y) in zip(tp, total_false)]
    f1_score = safe_per_element_division(tp, denominator)
    return f1_score


def per_class_accuracy(tp, tn, total_examples):
    total_correct = [x + y for (x, y) in zip(tp, tn)]
    accuracy = [x / total_examples for x in total_correct]
    return accuracy


def per_class_stats(tp, tn, fp, fn, total_examples):
    total_positives = [x + y for (x, y) in zip(tp, fn)]
    total_negatives = [x + y for (x, y) in zip(tn, fp)]
    accuracy = per_class_accuracy(tp, tn, total_examples)
    tp_rate = safe_per_element_division(tp, total_positives)
    tn_rate = safe_per_element_division(tn, total_negatives)
    f1_score = per_class_f1_score(tp, fp, fn)
    return accuracy, tp_rate, tn_rate, f1_score


def sensitivity(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def specificity(tn, fp):
    if tn + fp == 0:
        return 0
    return tn / (tn + fp)

