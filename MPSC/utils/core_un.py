import torch
from tqdm import tqdm
from utils.meter import AverageMeter
from utils.module_cslt import byx_getscores
import numpy as np


def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    """
    Computes the precision@k for the specified values of k in this mini-batch
    :param y_pred   : tensor, shape -> (batch_size, n_classes)
    :param y_actual : tensor, shape -> (batch_size)
    :param topk     : tuple
    :param return_tensor : bool, whether to return a tensor or a scalar
    :return:
        list, each element is a tensor with shape torch.Size([])
    """
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def evaluate(dataloader, model, dev, evaluation, topk=(1,)):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :return:     [list[float]]    topk accuracy
    """
    model.eval()
    test_loss = AverageMeter()
    test_loss.reset()
    preds = []
    scores = []

    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            x = sample['data'].to(dev)
            y = sample['label'].to(dev)
            output = model(x)
            logits = output['logits']
            type_prob = output['weights']
            ood_pred_prob = type_prob[:,-1].cpu().numpy()
            scores.append(ood_pred_prob)
            idx = y >= 0
            if len(logits[idx])!=0:
                loss = torch.nn.functional.cross_entropy(logits[idx], y[idx])
                test_loss.update(loss.item(), x.size(0))
            pred = get_pred(logits, topk).cpu().numpy()
            preds.append(pred)

            # acc = accuracy(logits, y, topk)

        preds = np.squeeze(np.concatenate(preds))
        scores = -np.concatenate(scores)
        acc = evaluation.close_accuracy(preds)
        open_detection = evaluation.open_detection_indexes(scores, -9999999)
        open_reco = evaluation.open_reco_indexes(scores, -9999999, preds)
        auroc = open_detection['auroc']
        macro_f1 = open_reco['macro_f1']

    return {'accuracy': acc, 'loss': test_loss.avg, 'auroc': auroc, 'macro_f1': macro_f1}

def get_pred(y_pred, topk=(1, )):
    """
    Computes the precision@k for the specified values of k in this mini-batch
    :param y_pred   : tensor, shape -> (batch_size, n_classes)
    :param y_actual : tensor, shape -> (batch_size)
    :param topk     : tuple
    :param return_tensor : bool, whether to return a tensor or a scalar
    :return:
        list, each element is a tensor with shape torch.Size([])
    """
    maxk = max(topk)

    _, pred = y_pred.topk(maxk, 1, True, True)

    return pred