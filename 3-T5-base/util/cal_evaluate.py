import rouge
import numpy as np

def compute_rouge(label, pred, weights=None, mode='weighted'):
    rouge_cal = rouge.Rouge()
    weights = weights or (0.2, 0.4, 0.4)
    if isinstance(label, str):
        label = [label]
    if isinstance(pred, str):
        pred = [pred]
    label = [' '.join(x) for x in label]
    pred = [' '.join(x) for x in pred]

    def _compute_rouge(label, pred):
        try:
            scores = rouge_cal.get_scores(hyps=label, refs=pred)[0]
            scores = [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]
        except ValueError:
            scores = [0, 0, 0]
        return scores

    scores = np.mean([_compute_rouge(*x) for x in zip(label, pred)], axis=0)
    if mode == 'weighted':
        return {'rouge': sum(s * w for s, w in zip(scores, weights))}
    elif mode == '1':
        return {'rouge-1': scores[0]}
    elif mode == '2':
        return {'rouge-2':scores[1]}
    elif mode == 'l':
        return {'rouge-l': scores[2]}
    elif mode == 'all':
        return {'rouge-1': scores[0], 'rouge-2':scores[1], 'rouge-l': scores[2]}