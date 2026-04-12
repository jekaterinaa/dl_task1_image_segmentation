import torch


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def compute_confusion_per_class(preds, targets, num_classes):
    """
    preds, targets: (N,H,W) int64 tensors
    returns:
        tp, fp, fn, tn: each (num_classes,) tensors
    """
    preds   = preds.view(-1)
    targets = targets.view(-1)
    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)
    tn = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        pred_c = preds == c
        targ_c = targets == c
        tp[c] = torch.sum(pred_c & targ_c)
        fp[c] = torch.sum(pred_c & ~targ_c)
        fn[c] = torch.sum(~pred_c & targ_c)
        tn[c] = torch.sum(~pred_c & ~targ_c)
    return tp, fp, fn, tn


def compute_metrics(tp, fp, fn, tn, eps=1e-8):
    """
    tp, fp, fn, tn: (num_classes,)
    returns:
        accuracy, precision, recall, f1: (num_classes,)
    """
    accuracy  = (tp + tn).float() / (tp + fp + fn + tn).float().clamp(min=1)
    precision = tp.float() / (tp + fp).float().clamp(min=1)
    recall    = tp.float() / (tp + fn).float().clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return accuracy, precision, recall, f1