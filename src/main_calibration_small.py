import abc
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
import os
import gc
from pathlib import Path
from data.data_utils import load_data
from model.model import create_model
from calibloss import \
    NodewiseECE, NodewiseBrier, NodewiseNLL, Reliability, NodewiseKDE, \
    NodewiswClassECE
from utils import \
    set_global_seeds, arg_parse, name_model, create_nested_defaultdict, \
    metric_mean, metric_std, default_cal_wdecay, default_cal_lr

import torch_sparse
from torch_sparse import SparseTensor
import numpy as np
from calibrator.simi_calibrator import Simi_Mailbox

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# collects metrics for evaluation
class Metrics(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def acc(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def nll(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def brier(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def ece(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def reliability(self) -> Reliability:
        raise NotImplementedError

    @abc.abstractmethod
    def kde(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def cls_ece(self) -> float:
        raise NotImplementedError

class NodewiseMetrics(Metrics):
    def __init__(
            self, logits: Tensor, gts: LongTensor, index: LongTensor,
            bins: int = 15, scheme: str = 'equal_width', norm=1):
        self.node_index = index
        self.logits = logits
        self.gts = gts
        self.nll_fn = NodewiseNLL(index)
        self.brier_fn = NodewiseBrier(index)
        self.ece_fn = NodewiseECE(index, bins, scheme, norm)
        self.kde_fn = NodewiseKDE(index, norm)
        self.cls_ece_fn = NodewiswClassECE(index, bins, scheme, norm)

    def acc(self) -> float:
        preds = torch.argmax(self.logits, dim=1)[self.node_index]
        return torch.mean(
            (preds == self.gts[self.node_index]).to(torch.get_default_dtype())
        ).item()

    def nll(self) -> float:
        return self.nll_fn(self.logits, self.gts).item()

    def brier(self) -> float:
        return self.brier_fn(self.logits, self.gts).item()

    def ece(self) -> float:
        return self.ece_fn(self.logits, self.gts).item()

    def reliability(self) -> Reliability:
        return self.ece_fn.get_reliability(self.logits, self.gts)
    
    def kde(self) -> float:
        return self.kde_fn(self.logits, self.gts).item()

    def cls_ece(self) -> float:
        return self.cls_ece_fn(self.logits, self.gts).item()


def evaluate(data, prob, mask_name):
    if mask_name == 'Train':
        mask = data.train_mask
    elif mask_name == 'Val':
        mask = data.val_mask
    elif mask_name == 'Test':
        mask = data.test_mask
    else:
        raise ValueError("Invalid mask_name")
    eval_result = {}
    eval = NodewiseMetrics(prob, data.y, mask)
    acc, nll, brier, ece, kde, cls_ece = eval.acc(), eval.nll(), \
                                eval.brier(), eval.ece(), eval.kde(), eval.cls_ece()
    eval_result.update({'acc':acc,
                        'nll':nll,
                        'bs':brier,
                        'ece':ece,
                        'kde':kde,
                        'cls_ece': cls_ece})
    reliability = eval.reliability()
    del eval
    gc.collect()
    return eval_result, reliability


def main(split, init, eval_type_list, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cal_train_result = create_nested_defaultdict(eval_type_list)
    cal_val_result = create_nested_defaultdict(eval_type_list)
    cal_test_result = create_nested_defaultdict(eval_type_list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        # Load data
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data
        values = torch.ones(data.edge_index.size(1))
        data.adj_t = SparseTensor.from_edge_index(data.edge_index, values, sparse_sizes=(data.num_nodes, data.num_nodes))
        data = data.to(device)

        # Load model
        model = create_model(dataset, args).to(device)
        model_name = name_model(fold, args)
        dir = Path(os.path.join('model', args.dataset, args.split_type, 'split'+str(split), 'init'+ str(init)))
        file_name = dir / (model_name + '.pt')
        model.load_state_dict(torch.load(file_name))
        torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.adj_t)

        ### Calibration
        temp_model = Simi_Mailbox(model, data, args)
        temp_model.compute_mailbox(logits, data)
        
        ### Train the calibrator on validation set
        cal_wdecay = args.cal_wdecay if args.cal_wdecay is not None else default_cal_wdecay(args)
        cal_lr = args.cal_lr if args.cal_lr is not None else default_cal_lr(args)
        temp_model.fit(data, data.val_mask, data.train_mask, cal_lr, cal_wdecay)
        
        with torch.no_grad():
            temp_model.eval()
            logits = temp_model(logits)
            log_prob = F.log_softmax(logits, dim=1)
            
        for eval_type in eval_type_list:
            eval_result, _ = evaluate(data, log_prob, 'Train')
            for metric in eval_result.keys():
                cal_train_result[eval_type][metric].append(eval_result[metric])
        print(f"Train Acc.: {eval_result['acc']*100:.2f}, Train NLL: {eval_result['nll']:.4f}, ",
              f"Train ECE: {eval_result['ece']*100:.2f}, Train CLS_ECE: {eval_result['cls_ece']*100:.2f}")

        for eval_type in eval_type_list:
            eval_result, _ = evaluate(data, log_prob, 'Val')
            for metric in eval_result.keys():
                cal_val_result[eval_type][metric].append(eval_result[metric])
        print(f"Val Acc.: {eval_result['acc']*100:.2f}, Val NLL: {eval_result['nll']:.4f}, ",
              f"Val ECE: {eval_result['ece']*100:.2f}, Val CLS_ECE: {eval_result['cls_ece']*100:.2f}")
        
        for eval_type in eval_type_list:
            eval_result, _ = evaluate(data, log_prob, 'Test')
            for metric in eval_result.keys():
                cal_test_result[eval_type][metric].append(eval_result[metric])
        print(f"Test Acc.: {eval_result['acc']*100:.2f}, Test NLL: {eval_result['nll']:.4f}, ",
              f"Test ECE: {eval_result['ece']*100:.2f}, Test CLS_ECE: {eval_result['cls_ece']*100:.2f}")
        print('---')
        
        torch.cuda.empty_cache()
    return cal_train_result, cal_val_result, cal_test_result


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    set_global_seeds(args.seed)
    eval_type_list = ['Nodewise']
    max_splits,  max_init = 5, 5

    cal_train_total = create_nested_defaultdict(eval_type_list)
    cal_val_total = create_nested_defaultdict(eval_type_list)
    cal_test_total = create_nested_defaultdict(eval_type_list)
    for split in range(max_splits):
        for init in range(max_init):
            print(split, init)
            (cal_train_result,
             cal_val_result,
             cal_test_result) = main(split, init, eval_type_list, args)
            for eval_type, eval_metric in cal_val_result.items():
                for metric in eval_metric:
                    cal_train_total[eval_type][metric].extend(cal_train_result[eval_type][metric])
                    cal_val_total[eval_type][metric].extend(cal_val_result[eval_type][metric])
                    cal_test_total[eval_type][metric].extend(cal_test_result[eval_type][metric])
            
    train_mean = metric_mean(cal_train_total['Nodewise'])
    val_mean = metric_mean(cal_val_total['Nodewise'])
    test_mean = metric_mean(cal_test_total['Nodewise'])
    test_std = metric_std(cal_test_total['Nodewise'])
        
    # print results
    for name, result in zip([args.calibration], [cal_test_total]):
        print(name)
        for eval_type in eval_type_list:
            test_mean = metric_mean(result[eval_type])
            test_std = metric_std(result[eval_type])
            print(f"{eval_type:>8} Accuracy: &{test_mean['acc']:.2f}$\pm${test_std['acc']:.2f} \t" + \
                                f"NLL: &{test_mean['nll']:.4f}$\pm${test_std['nll']:.4f} \t" + \
                                f"Brier: &{test_mean['bs']:.4f}$\pm${test_std['bs']:.4f} \t" + \
                                f"ECE: &{test_mean['ece']:.2f}$\pm${test_std['ece']:.2f} \t" + \
                                f"Classwise-ECE: &{test_mean['cls_ece']:.2f}$\pm${test_std['cls_ece']:.2f} \t" + \
                                f"KDE: &{test_mean['kde']:.2f}$\pm${test_std['kde']:.2f}")