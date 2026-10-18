import numpy as np

from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc


# the below function is taken from OmniAnomaly code base directly
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def _clean_scores(*arrays):
    cleaned = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        cleaned.append(arr)
    return cleaned


def _fallback_threshold(init_score, score):
    baseline = init_score if init_score.size else score
    if baseline.size == 0:
        return 0.0
    threshold = np.percentile(baseline, 100 * lm[0]) * lm[1]
    if not np.isfinite(threshold):
        threshold = np.max(baseline)
    return threshold


def pot_eval(init_score, score, label, q=1e-5, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """
    init_score = np.asarray(init_score, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label).reshape(-1)
    finite_mask = np.isfinite(score)
    score = score[finite_mask]
    label = label[finite_mask]
    init_score, = _clean_scores(init_score)

    if init_score.size == 0 or score.size == 0:
        pot_th = _fallback_threshold(init_score, score)
        pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
        p_t = calc_point2point(pred, label)
        return {
            'f1': p_t[0], 'precision': p_t[1], 'recall': p_t[2],
            'TP': p_t[3], 'TN': p_t[4], 'FP': p_t[5], 'FN': p_t[6],
            'ROC/AUC': p_t[7], 'threshold': pot_th,
        }, np.array(pred)

    s = None
    last_error = None
    levels = [lm[0], 0.999, 0.995, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    levels = [x for i, x in enumerate(levels) if 0 < x < 1 and x not in levels[:i]]
    for lms in levels:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
        except (ArithmeticError, FloatingPointError, IndexError, ValueError) as exc:
            last_error = exc
        else:
            break
    else:
        pot_th = _fallback_threshold(init_score, score)
        pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
        p_t = calc_point2point(pred, label)
        return {
            'f1': p_t[0], 'precision': p_t[1], 'recall': p_t[2],
            'TP': p_t[3], 'TN': p_t[4], 'FP': p_t[5], 'FN': p_t[6],
            'ROC/AUC': p_t[7], 'threshold': pot_th,
        }, np.array(pred)

    ret = s.run(dynamic=False)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    thresholds = np.asarray(ret['thresholds'], dtype=np.float64)
    thresholds = thresholds[np.isfinite(thresholds)]
    pot_th = np.mean(thresholds) * lm[1] if thresholds.size else _fallback_threshold(init_score, score)
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    # DEBUG - np.save(f'{debug}.npy', np.array(pred))
    # DEBUG - print(np.argwhere(np.array(pred)))
    p_t = calc_point2point(pred, label)
    # print('POT result: ', p_t, pot_th, p_latency)
    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
        # 'pot-latency': p_latency
    }, np.array(pred)
