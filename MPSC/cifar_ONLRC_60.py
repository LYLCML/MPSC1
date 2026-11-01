import os
import sys
import pathlib
import time
import datetime
import argparse
import shutil
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torch.cuda.amp import autocast, GradScaler
from utils.core_un import accuracy, evaluate
from utils.builder_un import *
from utils.utils_un import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console
from utils.loss import *
from utils.module_cslt import CSGRLClassifier, byx_getscores, MLPHead
from utils.plotter import plot_results
from PIL import ImageFile
import utils.metrics

import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Compatibility for the legacy randaugment package with NumPy >= 1.24.
if not hasattr(np, 'int'):
    np.int = int

LOG_FREQ = 1

# Direct-run configuration (migrated from scripts/cifar_un.sh).
# Edit these values, then run this file directly on Windows.
RUN_CONFIG = {
    'config': 'config/cifar100.cfg',
    'synthetic_data': 'cifar60no',
    'noise_type': 'symmetric',
    'closeset_noise_ratio': 0.4,
    'gpu': '1',
    'net': 'cnn',
    'batch_size': 128,
    'lr': 0.001,
    'lr_decay': 'linear',
    'weight_decay': 1e-5,
    'opt': 'adam',
    'warmup_epochs': 10,
    'warmup_lr_scale': 1.0,
    'epochs': 200,
    'save_model': True,
    'use_fp16': True,
    'use_grad_accumulate': False,
    'project': '',
    'log': 'ONLRC_mutual_three_proto_entropy_ood_sharedR_lr0.001',
    'epsilon': 0.6,
    'eta': 0.5,
    'alpha': 0.5,
    'beta': 0,
    'omega': 1,
    'decay_epoch': 80,
    'seed': 0,
    'rho': 1.0,
    'loss_func_aux': 'mae',
    'neg_cons': False,
    'activation': 'relu',
    'ablation': False,
    'proto_radius': 1.0,
    'proto_temperature': 1.0,
    'proto_lambda': 0,
    'mutual_lambda': 1.0,
    'mutual_ramp_epochs': 10,
    'peer_temperature': 1,
}

# OOD and IDN independently search the same discrete temperature range.
# The previously unstable 0.01 and 0.02 values are intentionally excluded.
ROUTE_TEMPERATURE_CANDIDATES = (0.5000,  0.05)
ROUTE_TEMPERATURE_SEARCH_START_EPOCH = 999


class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak(sample)
        x_w2 = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w1, x_w2, x_s


class PrototypeClassifier(nn.Module):
    """Euclidean prototype classifier with a soft radius constraint."""

    def __init__(self, feat_dim, num_classes, radius=1.0,
                 temperature=1.0):
        super().__init__()
        prototypes = F.normalize(
            torch.randn(num_classes, feat_dim), dim=1
        ) * radius
        self.prototypes = nn.Parameter(prototypes)
        self.radius = nn.Parameter(torch.tensor(float(radius)))
        self.temperature = temperature

    def forward(self, features):
        # Squared Euclidean distances are sensitive to cancellation in FP16.
        # Keep this small B x C x D computation in FP32.
        with autocast(enabled=False):
            distances = (
                features.float().unsqueeze(1)
                - self.prototypes.float().unsqueeze(0)
            ).pow(2).sum(dim=2)
        logits = -distances / self.temperature
        return logits, distances

    def radius_loss(self):
        return (
            self.prototypes.norm(dim=1) - self.radius
        ).pow(2).mean()


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25,
                 momentum=0.1, activation='tanh', proto_radius=1.0,
                 proto_temperature=1.0):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Two independent high-level branches. Low-level edge/texture
        # extraction remains shared, while semantic errors need not be
        # identical in the two mutually supervising spaces.
        self.block3_b = copy.deepcopy(self.block3)
        self.block3_c = copy.deepcopy(self.block3)
        # Restore the representation capacity of the original classifier:
        # A 32x32 input becomes 128x4x4 here, i.e. a 2048-D flattened
        # representation. Keep the final embedding at 2048 dimensions;
        # classification is performed by Euclidean class prototypes.
        self.prototype_embedding = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )
        for module in self.prototype_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.classfier_head = PrototypeClassifier(
            feat_dim=2048,
            num_classes=n_outputs,
            radius=proto_radius,
            temperature=proto_temperature,
        )
        self.routing_embedding = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )
        for module in self.routing_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Space B is a complete class-prototype space, rather than an
        # (C + 1)-way routing head. The origin represents the OOD interior;
        # no OOD semantic prototype is placed on the known-class sphere.
        self.weight_head = PrototypeClassifier(
            feat_dim=2048,
            num_classes=n_outputs,
            radius=proto_radius,
            temperature=proto_temperature,
        )
        # Both prototype spaces lie on the same learnable radius R.
        self.weight_head.radius = self.classfier_head.radius
        self.prototype_embedding_c = copy.deepcopy(
            self.routing_embedding
        )
        self.classifier_head_c = PrototypeClassifier(
            feat_dim=2048,
            num_classes=n_outputs,
            radius=proto_radius,
            temperature=proto_temperature,
        )
        # All three prototype spaces share one learnable radius R.
        self.classifier_head_c.radius = self.classfier_head.radius

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x_a = self.block3(x)
        x_b = self.block3_b(x)
        x_c = self.block3_c(x)
        x_a = x_a.view(x_a.size(0), -1)
        x_b = x_b.view(x_b.size(0), -1)
        x_c = x_c.view(x_c.size(0), -1)
        features = self.prototype_embedding(x_a)
        logits, distances = self.classfier_head(features)
        routing_features = self.routing_embedding(x_b)
        logits_b, routing_distances = self.weight_head(routing_features)
        features_c = self.prototype_embedding_c(x_c)
        logits_c, distances_c = self.classifier_head_c(features_c)
        # The two spaces have independent coordinates. Averaging their class
        # evidence gives the prediction used by evaluation.
        ensemble_logits = (logits + logits_b + logits_c) / 3.0
        # Backward-compatible evaluation score. ``utils/core_un.py`` only
        # reads the last column of ``weights`` as the OOD probability.
        # Here it is produced by the new two-space geometry rather than by
        # the removed (C + 1)-way routing head.
        # Origin and the nearest known prototype compete in the same
        # dimensionless Euclidean-distance space.
        relative_origin_distance_a = (
            features.float().norm(dim=1)
            / self.classfier_head.radius.abs().clamp_min(1e-4)
        )
        relative_origin_distance_b = (
            routing_features.float().norm(dim=1)
            / self.weight_head.radius.abs().clamp_min(1e-4)
        )
        relative_origin_distance_c = (
            features_c.float().norm(dim=1)
            / self.classifier_head_c.radius.abs().clamp_min(1e-4)
        )
        origin_distance = (
            relative_origin_distance_a
            + relative_origin_distance_b
            + relative_origin_distance_c
        ) / 3.0
        radius_sq_a = self.classfier_head.radius.abs().clamp_min(
            1e-4
        ).pow(2)
        radius_sq_b = self.weight_head.radius.abs().clamp_min(
            1e-4
        ).pow(2)
        nearest_proto_distance_a = (
            distances.min(dim=1).values / radius_sq_a
        )
        nearest_proto_distance_b = (
            routing_distances.min(dim=1).values / radius_sq_b
        )
        radius_sq_c = self.classifier_head_c.radius.abs().clamp_min(
            1e-4
        ).pow(2)
        nearest_proto_distance_c = (
            distances_c.min(dim=1).values / radius_sq_c
        )
        relative_proto_distance = (
            nearest_proto_distance_a
            + nearest_proto_distance_b
            + nearest_proto_distance_c
        ) / 3.0
        # Select the maximum-probability class in each prototype space and
        # use its raw feature-prototype dot product. No normalization is
        # applied; a smaller dot product indicates a more OOD-like sample.
        pred_a = logits.argmax(dim=1)
        pred_b = logits_b.argmax(dim=1)
        pred_c = logits_c.argmax(dim=1)
        dot_a = (
            features.float()
            * self.classfier_head.prototypes[pred_a].float()
        ).sum(dim=1)
        dot_b = (
            routing_features.float()
            * self.weight_head.prototypes[pred_b].float()
        ).sum(dim=1)
        dot_c = (
            features_c.float()
            * self.classifier_head_c.prototypes[pred_c].float()
        ).sum(dim=1)
        mean_max_class_dot = (dot_a + dot_b + dot_c) / 3.0
        ood_prob = torch.sigmoid(-mean_max_class_dot)
        # Only the final column is consumed as the open-set score by the
        # evaluator; keep the class columns as ordinary probabilities.
        known_class_prob = ensemble_logits.softmax(dim=1)
        weights = torch.cat(
            (known_class_prob, ood_prob.unsqueeze(1)), dim=1
        )

        return {
            'logits': ensemble_logits,
            'logits_a': logits,
            'logits_b': logits_b,
            'logits_c': logits_c,
            'distances': distances,
            'features': features,
            'backbone_features': x_a,
            'backbone_features_b': x_b,
            'backbone_features_c': x_c,
            'routing_features': routing_features,
            'routing_distances': routing_distances,
            'features_c': features_c,
            'distances_c': distances_c,
            'origin_distance': origin_distance,
            'relative_proto_distance': relative_proto_distance,
            'weights': weights,
            's_logits': logits_b,
        }


def prototype_type_prob(output, weak_output, labels, openset_ratio,
                        closeset_noise_ratio, eta=0.5,
                        ood_temperature=0.05,
                        idn_temperature=0.5,
                        previous_ood_quantiles=None,
                        previous_idn_quantiles=None):
    """Return per-space [clean, closed-set-noise, OOD] probabilities.

    OOD evidence combines normalized entropy and weak-view disagreement.
    Except during the first epoch, routing uses fixed quantile thresholds
    computed from every training sample in the preceding epoch.
    """
    # Some dataset variants return labels as int32. PyTorch gather/scatter
    # require int64 indices.
    labels = labels.long()
    results = []
    for space_index, (logits, weak_logits) in enumerate((
        (output['logits_a'], weak_output['logits_a']),
        (output['logits_b'], weak_output['logits_b']),
        (output['logits_c'], weak_output['logits_c']),
    )):
        class_prob = 0.5 * (
            logits.softmax(dim=1) + weak_logits.softmax(dim=1)
        )
        label_prob = class_prob.gather(1, labels[:, None]).squeeze(1)
        alternative = class_prob.clone()
        alternative.scatter_(1, labels[:, None], -1.0)
        alternative_prob = alternative.max(dim=1).values

        original_prob = logits.softmax(dim=1).clamp_min(1e-8)
        weak_prob = weak_logits.softmax(dim=1).clamp_min(1e-8)
        normalized_entropy = -(
            original_prob * original_prob.log()
        ).sum(dim=1) / float(np.log(original_prob.size(1)))
        disagreement = js_div(original_prob, weak_prob)
        ood_evidence = (
            (1.0 - eta) * normalized_entropy
            + eta * disagreement
        )
        if previous_ood_quantiles is None:
            # Only the first training epoch has no previous-epoch evidence.
            ood_quantile = torch.quantile(
                ood_evidence.detach(), 1.0 - float(openset_ratio)
            )
        else:
            ood_quantile = previous_ood_quantiles[space_index].to(
                device=ood_evidence.device, dtype=ood_evidence.dtype
            )
        ood_score = torch.sigmoid(
            (ood_evidence - ood_quantile)
            / max(float(ood_temperature), 1e-6)
        )
        known_prob = 1.0 - ood_score

        # Within the known portion, closed-set noise is the tail whose best
        # alternative class most strongly beats the observed label. A second
        # quantile calibration prevents ordinary hard clean samples from
        # gradually being absorbed into IDN.
        idn_evidence = (
            alternative_prob.clamp_min(1e-8).log()
            - label_prob.clamp_min(1e-8).log()
        )
        # closeset_noise_ratio is conditional on known classes in the
        # dataset transition matrix (e.g. 20% of the 80% known portion).
        conditional_idn_ratio = min(float(closeset_noise_ratio), 0.95)
        total_idn_ratio = (
            (1.0 - float(openset_ratio)) * conditional_idn_ratio
        )
        if previous_idn_quantiles is None:
            idn_quantile = torch.quantile(
                idn_evidence.detach(), 1.0 - total_idn_ratio
            )
        else:
            idn_quantile = previous_idn_quantiles[space_index].to(
                device=idn_evidence.device, dtype=idn_evidence.dtype
            )
        idn_conditional = torch.sigmoid(
            (idn_evidence - idn_quantile)
            / max(float(idn_temperature), 1e-6)
        )
        clean_conditional = 1.0 - idn_conditional
        clean_score = known_prob * clean_conditional
        idn_score = known_prob * idn_conditional
        type_prob = torch.stack(
            (clean_score, idn_score, ood_score), dim=1
        )
        type_prob = type_prob / type_prob.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        results.append((
            type_prob, class_prob,
            ood_evidence.detach(), idn_evidence.detach()
        ))
    return results


@torch.no_grad()
def search_route_temperatures(ood_evidence, idn_evidence,
                              ood_quantiles, idn_quantiles,
                              openset_ratio, closeset_noise_ratio,
                              candidates=ROUTE_TEMPERATURE_CANDIDATES):
    """Select OOD first, then IDN temperature from full-epoch evidence."""
    target_ood = float(openset_ratio)
    target_idn = (
        (1.0 - target_ood) * min(float(closeset_noise_ratio), 0.95)
    )
    ood_delta = ood_evidence.float() - ood_quantiles.float()[:, None]
    idn_delta = idn_evidence.float() - idn_quantiles.float()[:, None]

    ood_temperature = min(
        candidates,
        key=lambda value: abs(
            torch.sigmoid(ood_delta / float(value)).mean().item()
            - target_ood
        )
    )
    ood_prob = torch.sigmoid(ood_delta / float(ood_temperature))
    known_prob = 1.0 - ood_prob

    idn_temperature = min(
        candidates,
        key=lambda value: abs(
            (
                known_prob
                * torch.sigmoid(idn_delta / float(value))
            ).mean().item() - target_idn
        )
    )
    predicted_ood = ood_prob.mean().item()
    predicted_idn = (
        known_prob * torch.sigmoid(idn_delta / float(idn_temperature))
    ).mean().item()
    predicted_clean = 1.0 - predicted_ood - predicted_idn
    return (
        float(ood_temperature), float(idn_temperature),
        (predicted_clean, predicted_idn, predicted_ood)
    )


def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def record_network_arch(result_dir, net):
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())


def get_smoothed_label_distribution(labels, num_class, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))
    label_indices = torch.unsqueeze(labels, dim=1).long().cpu()
    smoothed_label.scatter_(dim=1, index=label_indices, value=1 - epsilon)
    return smoothed_label.to(labels.device)


def build_logger(params):
    logger_root = f'Results/{params.synthetic_data}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_noise_ratio * 100)
    noise_condition = f'symm_{percentile:2d}' if params.noise_type == 'symmetric' else f'asym_{percentile:2d}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, noise_condition, params.project, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir


def build_model_optim_scheduler(params, device, build_scheduler=True):
    assert params.dataset.startswith('cifar')
    n_classes = int(params.n_classes * (1 - params.openset_ratio))
    net = CNN(input_channel=3, n_outputs=n_classes,
              activation='leaky relu' if params.activation == 'l_relu' else params.activation,
              proto_radius=params.proto_radius,
              proto_temperature=params.proto_temperature)
    if params.opt == 'sgd':
        optimizer = build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    elif params.opt == 'adam':
        optimizer = build_adam_optimizer(net.parameters(), params.lr)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet.')
    if build_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,
                                                               verbose=True, threshold=1e-4)
    else:
        scheduler = None
    return net.to(device), optimizer, scheduler, n_classes


def build_lr_plan(params, factor=10, decay='linear'):
    epoch_decay_start = params.decay_epoch
    lr_plan = [params.lr] * params.epochs
    for i in range(0, params.warmup_epochs):
        lr_plan[i] *= factor
    for i in range(epoch_decay_start, params.epochs):
        if decay == 'linear':
            lr_plan[i] = float(params.epochs - i) / (params.epochs - params.warmup_epochs) * params.lr  # linearly decay
        elif decay == 'cosine':
            lr_plan[i] = 0.5 * params.lr * (1 + math.cos(
                (i - params.warmup_epochs + 1) * math.pi / (params.epochs - params.warmup_epochs + 1)))  # cosine decay
        else:
            raise AssertionError(f'lr decay method: {decay} is not implemented yet.')
    return lr_plan


def build_dataset_loader(params):
    assert params.dataset.startswith('cifar')
    # randaugment contains lambdas that cannot be pickled by Windows' spawn
    # multiprocessing method. Linux can keep using parallel loader workers.
    num_workers = 0 if os.name == 'nt' else 8
    transform = build_transform(rescale_size=params.rescale_size, crop_size=params.crop_size)
    if params.dataset == 'cifar100':
        dataset = build_cifar100n_dataset(os.path.join(params.database, params.dataset),
                                          CLDataTransform(transform['cifar_train'],
                                                          transform['cifar_train_strong_aug']), transform['cifar_test'],
                                          noise_type=params.noise_type, openset_ratio=params.openset_ratio,
                                          closeset_noise_ratio=params.closeset_noise_ratio)
    else:
        raise AssertionError(f'{params.dataset} dataset is not supported yet.')
    train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=False)
    return dataset, train_loader, test_loader


def wrapup_training(result_dir, best_macro_f1):
    stats = get_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f'{result_dir}-bestF1_{best_macro_f1:.4f}')


def main(cfg, device):
    init_seeds(cfg.seed)
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16

    logger, result_dir = build_logger(cfg)
    net, optimizer, scheduler, n_classes = build_model_optim_scheduler(cfg, device, build_scheduler=False)
    lr_plan = build_lr_plan(cfg, factor=cfg.warmup_lr_scale, decay=cfg.lr_decay)
    dataset, train_loader, test_loader = build_dataset_loader(cfg)
    evaluation = utils.metrics.OSREvaluation(test_loader)
    n_train_samples = dataset['n_train_samples']
    # Per-sample evidence from the current epoch. Its full-dataset quantiles
    # become the fixed routing thresholds used by the following epoch.
    epoch_ood_evidence = torch.empty(3, n_train_samples)
    epoch_idn_evidence = torch.empty(3, n_train_samples)
    previous_ood_quantiles = None
    previous_idn_quantiles = None
    # Both start at 0.1. They are selected once after epoch 20 and then frozen.
    route_ood_temperature = 0.05
    route_idn_temperature = 0.5
    diagnostic_true_type = torch.empty(
        n_train_samples, dtype=torch.long
    )
    diagnostic_idn, diagnostic_ood, diagnostic_clean = (
        dataset['train'].get_sets()
    )
    diagnostic_true_type[diagnostic_clean] = 0
    diagnostic_true_type[diagnostic_idn] = 1
    diagnostic_true_type[diagnostic_ood] = 2

    logger.msg(
        f"Categories: {n_classes}, Training Samples: {dataset['n_train_samples']}, Testing Samples: {dataset['n_test_samples']}")
    logger.msg(
        f"Noise Type: {dataset['train'].noise_type}, Openset Noise Ratio: {dataset['train'].openset_noise_ratio}, Closedset Noise Ratio: {dataset['train'].closeset_noise_rate}")
    logger.msg(f'Optimizer: {cfg.opt}')
    record_network_arch(result_dir, net)

    if cfg.loss_func_aux == 's-mae':
        aux_loss_func = F.smooth_l1_loss
    elif cfg.loss_func_aux == 'mae':
        aux_loss_func = F.l1_loss
    elif cfg.loss_func_aux == 'mse':
        aux_loss_func = F.mse_loss
    else:
        raise AssertionError(f'{cfg.loss_func_aux} loss is not supported for auxiliary loss yet.')

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    train_clean_prob = AverageMeter()
    train_idn_prob = AverageMeter()
    train_ood_prob = AverageMeter()
    train_space_agreement = AverageMeter()
    true_radius_meters = [AverageMeter() for _ in range(3)]
    epoch_train_time = AverageMeter()
    best_macro_f1, best_epoch = 0.0, None
    scaler = GradScaler()
    iters_to_accumulate = round(64 / cfg.batch_size) if cfg.use_grad_accumulate and cfg.batch_size < 64 else 1
    logger.msg(
        f'Accumulate gradients every {iters_to_accumulate} iterations --> Acutal batch size is {cfg.batch_size * iters_to_accumulate}')

    entropy_normalize_factor = entropy(torch.ones(n_classes) / n_classes).item()
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, cfg.epochs):
        start_time = time.time()

        net.train()
        adjust_lr(optimizer, lr_plan[epoch])
        optimizer.zero_grad()
        train_loss.reset()
        train_accuracy.reset()
        train_clean_prob.reset()
        train_idn_prob.reset()
        train_ood_prob.reset()
        train_space_agreement.reset()
        evidence_seen = torch.zeros(n_train_samples, dtype=torch.bool)
        for meter in true_radius_meters:
            meter.reset()

        # train this epoch
        pbar = tqdm(train_loader, ncols=150, ascii=' >', leave=False, desc='training')
        for it, sample in enumerate(pbar):
            curr_lr = [group['lr'] for group in optimizer.param_groups][0]
            # torch.autograd.set_detect_anomaly(True)

            s = time.time()

            indices = sample['index']
            x, x_w, x_s = sample['data']
            x, x_w, x_s = x.to(device), x_w.to(device), x_s.to(device)
            y = sample['label'].to(device)

            with autocast(cfg.use_fp16):
                output = net(x)
                logits = output['logits']
                probs = logits.softmax(dim=1)
                # er += torch.abs(er.min())
                # er /= torch.max(er)
                train_acc = accuracy(logits, y, topk=(1,))

                output_s = net(x_s)
                logits_s = output_s['logits']
                output_w = net(x_w)
                logits_w = output_w['logits']
                space_types = prototype_type_prob(
                    output, output_w, y, cfg.openset_ratio,
                    cfg.closeset_noise_ratio, eta=cfg.eta,
                    ood_temperature=route_ood_temperature,
                    idn_temperature=route_idn_temperature,
                    previous_ood_quantiles=previous_ood_quantiles,
                    previous_idn_quantiles=previous_idn_quantiles
                )
                type_a, class_a, ood_evidence_a, idn_evidence_a = (
                    space_types[0]
                )
                type_b, class_b, ood_evidence_b, idn_evidence_b = (
                    space_types[1]
                )
                type_c, class_c, ood_evidence_c, idn_evidence_c = (
                    space_types[2]
                )
                memory_indices = indices.long().cpu()
                epoch_ood_evidence[:, memory_indices] = torch.stack((
                    ood_evidence_a.float().cpu(),
                    ood_evidence_b.float().cpu(),
                    ood_evidence_c.float().cpu(),
                ))
                epoch_idn_evidence[:, memory_indices] = torch.stack((
                    idn_evidence_a.float().cpu(),
                    idn_evidence_b.float().cpu(),
                    idn_evidence_c.float().cpu(),
                ))
                evidence_seen[memory_indices] = True
                type_prob = (type_a + type_b + type_c) / 3.0
                pred_a = class_a.argmax(dim=1)
                pred_b = class_b.argmax(dim=1)
                pred_c = class_c.argmax(dim=1)
                space_agreement = (
                    (pred_a == pred_b).float()
                    + (pred_a == pred_c).float()
                    + (pred_b == pred_c).float()
                ) / 3.0
                clean_pred_prob = type_prob[:, 0]
                idn_pred_prob = type_prob[:, 1]
                ood_pred_prob = type_prob[:, 2]

                pbar.set_postfix_str(f'TrainAcc: {train_accuracy.avg:3.2f}%; TrainLoss: {train_loss.avg:3.2f}')
                given_labels = get_smoothed_label_distribution(y, n_classes, epsilon=cfg.epsilon)
                # y_hot = torch.nn.functional.one_hot(y, n_classes)
                if epoch < cfg.warmup_epochs:
                    pbar.set_description(f'WARMUP TRAINING (lr={curr_lr:.3e})')
                    # Both prototype spaces learn the observed known label,
                    # using two independent weak views.
                    loss = (
                        cross_entropy(output['logits_a'], given_labels)
                        + cross_entropy(output_w['logits_a'], given_labels)
                        + cross_entropy(output['logits_b'], given_labels)
                        + cross_entropy(output_w['logits_b'], given_labels)
                        + cross_entropy(output['logits_c'], given_labels)
                        + cross_entropy(output_w['logits_c'], given_labels)
                    ) / 6.0
                    loss += cfg.proto_lambda * (
                        net.classfier_head.radius_loss()
                        + net.weight_head.radius_loss()
                        + net.classifier_head_c.radius_loss()
                    ) / 3.0
                else:
                    pbar.set_description(f'ROBUST TRAINING (lr={curr_lr:.3e})')
                    # Do not switch abruptly from supervised warm-up to fully
                    # self-routed learning. Early routing errors otherwise
                    # reinforce themselves and classify nearly everything as
                    # OOD in one epoch.
                    robust_epoch = epoch - cfg.warmup_epochs + 1
                    route_ramp = min(
                        1.0,
                        robust_epoch / float(cfg.mutual_ramp_epochs)
                    )

                    # strong aug, "NeurIPS 2020 - Unsupervised Data Augmentation for Consistency Training"
                    def space_loss(logits_0, logits_weak, logits_strong,
                                   strong_features, peer_mean, peer_type):
                        prob_0 = logits_0.softmax(dim=1)
                        prob_w = logits_weak.softmax(dim=1)
                        with torch.no_grad():
                            # The other space supplies the corrected class
                            # target. Power sharpening preserves probability
                            # ordering and is the standard temperature form.
                            sharp = peer_mean.clamp_min(1e-8).pow(
                                1.0 / cfg.peer_temperature
                            )
                            sharp = sharp / sharp.sum(
                                dim=1, keepdim=True
                            ).clamp_min(1e-8)
                        clean = 0.5 * (
                            cross_entropy(
                                logits_0, given_labels, reduction='none'
                            )
                            + cross_entropy(
                                logits_weak, given_labels, reduction='none'
                            )
                        )
                        clean += route_ramp * cfg.alpha * 0.5 * (
                            entropy_loss(logits_0, reduction='none')
                            + entropy_loss(
                                logits_weak, reduction='none'
                            )
                        )
                        idn = cfg.rho * cross_entropy(
                            logits_strong, sharp, reduction='none'
                        )
                        # Minimize ||z_s||_2 / D to pull OOD strong-view
                        # embeddings toward the origin. Dividing by the
                        # feature dimension keeps its scale independent of D.
                        origin_distance = strong_features.float().norm(
                            p=2, dim=1
                        )
                        ood = cfg.beta * (
                            origin_distance / float(strong_features.size(1))
                        )
                        # Routing uses only the detached preceding teacher in
                        # the cycle, preventing direct gradient shortcuts.
                        peer = peer_type.detach()
                        supervised_route = torch.zeros_like(peer)
                        supervised_route[:, 0] = 1.0
                        peer = (
                            (1.0 - route_ramp) * supervised_route
                            + route_ramp * peer
                        )
                        routed = (
                            peer[:, 0] * clean
                            + peer[:, 1] * idn
                            + peer[:, 2] * ood
                        ).mean()
                        cons = (
                            symmetric_kl_div(
                                prob_0.clamp_min(1e-8),
                                prob_w.clamp_min(1e-8)
                            )
                            * (peer[:, 0] + peer[:, 1])
                        ).mean()
                        return routed, cons

                    # Cyclic class targets: C -> A, A -> B, B -> C.
                    peer_mean_a = 0.5 * (
                        output['logits_c'].softmax(dim=1)
                        + output_w['logits_c'].softmax(dim=1)
                    )
                    peer_mean_b = 0.5 * (
                        output['logits_a'].softmax(dim=1)
                        + output_w['logits_a'].softmax(dim=1)
                    )
                    peer_mean_c = 0.5 * (
                        output['logits_b'].softmax(dim=1)
                        + output_w['logits_b'].softmax(dim=1)
                    )
                    peer_type_a = type_c
                    peer_type_b = type_a
                    peer_type_c = type_b

                    loss_a, cons_a = space_loss(
                        output['logits_a'], output_w['logits_a'],
                        output_s['logits_a'],
                        output_s['features'],
                        peer_mean_a, peer_type_a
                    )
                    loss_b, cons_b = space_loss(
                        output['logits_b'], output_w['logits_b'],
                        output_s['logits_b'],
                        output_s['routing_features'],
                        peer_mean_b, peer_type_b
                    )
                    loss_c, cons_c = space_loss(
                        output['logits_c'], output_w['logits_c'],
                        output_s['logits_c'],
                        output_s['features_c'],
                        peer_mean_c, peer_type_c
                    )

                    # Cyclic type alignment with detached teachers:
                    # KL(C||A) + KL(A||B) + KL(B||C).
                    mutual_type = (
                        F.kl_div(
                            type_a.clamp_min(1e-8).log(),
                            peer_type_a.detach(), reduction='batchmean'
                        )
                        + F.kl_div(
                            type_b.clamp_min(1e-8).log(),
                            peer_type_b.detach(), reduction='batchmean'
                        )
                        + F.kl_div(
                            type_c.clamp_min(1e-8).log(),
                            peer_type_c.detach(), reduction='batchmean'
                        )
                    ) / 3.0
                    loss = (
                        (loss_a + loss_b + loss_c) / 3.0
                        + route_ramp * cfg.omega
                        * (cons_a + cons_b + cons_c) / 3.0
                        + route_ramp * cfg.mutual_lambda * mutual_type
                    )
                    loss += cfg.proto_lambda * (
                        net.classfier_head.radius_loss()
                        + net.weight_head.radius_loss()
                        + net.classifier_head_c.radius_loss()
                    ) / 3.0

                    # debug
                    # loss_content = f'loss: {loss.item():3.2f}; loss_cls: (clean={loss_clean.mean():3.2f}, idn={loss_idn.mean():3.2f}, ood={loss_ood.mean():3.5f}); loss_aux: (clean={loss_aux_clean:3.2f}, ood={loss_aux_ood:3.2f})'
                    # with open(f'{result_dir}/debug-loss.txt', 'a') as f:
                    #     f.write(f'{loss_content}\n')
                    # prob_content = f'prob_clean: {clean_pred_prob.mean().item():3.5f}({clean_probs.mean().item():3.5f}); prob_id: {idn_pred_prob.mean().item():3.5f}; prob_ood: {ood_pred_prob.mean().item():3.5f}({ood_probs.mean().item():3.5f})'
                    # with open(f'{result_dir}/debug-prob.txt', 'a') as f:
                    #     f.write(f'{prob_content}\n')

            # if iters_to_accumulate > 0:
            #     loss = loss / iters_to_accumulate  # ???

            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                try:
                    scaler.step(optimizer)
                except RuntimeError:  # in case of "RuntimeError: Function 'CudnnBatchNormBackward' returned nan values in its 0th output."
                    logger.msg('Runtime Error occured! Have unscaled losses and clipped grads before optimizing!')
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_accuracy.update(train_acc[0], x.size(0))
            train_loss.update(loss.item(), x.size(0))
            train_clean_prob.update(
                clean_pred_prob.detach().mean().item(), x.size(0)
            )
            train_idn_prob.update(
                idn_pred_prob.detach().mean().item(), x.size(0)
            )
            train_ood_prob.update(
                ood_pred_prob.detach().mean().item(), x.size(0)
            )
            train_space_agreement.update(
                space_agreement.mean().item(), x.size(0)
            )
            batch_true_type = diagnostic_true_type[memory_indices]
            batch_radius = output[
                'origin_distance'
            ].detach().float().cpu()
            for type_index in range(3):
                type_mask = batch_true_type == type_index
                if type_mask.any():
                    true_radius_meters[type_index].update(
                        batch_radius[type_mask].mean().item(),
                        int(type_mask.sum())
                    )
            epoch_train_time.update(time.time() - s, 1)
            if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                total_mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                mem = torch.cuda.memory_reserved() / 2 ** 30
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                  f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                  f"Train Accuracy:[{train_accuracy.avg:6.2f}]  " \
                                  f"Loss:[{train_loss.avg:4.4f}]  " \
                                  f"GPU-MEM:[{mem:6.3f}/{total_mem:6.3f} Gb]  " \
                                  f"{epoch_train_time.avg:6.2f} sec/iter"
                logger.debug(console_content)

        if not evidence_seen.all():
            missing_count = int((~evidence_seen).sum())
            raise RuntimeError(
                f'{missing_count} training samples were not seen while '
                'building the epoch-wide routing thresholds.'
            )
        previous_ood_quantiles = torch.quantile(
            epoch_ood_evidence,
            1.0 - float(cfg.openset_ratio), dim=1
        )
        previous_idn_quantiles = torch.quantile(
            epoch_idn_evidence,
            1.0 - (
                (1.0 - float(cfg.openset_ratio))
                * min(float(cfg.closeset_noise_ratio), 0.95)
            ),
            dim=1
        )
        used_ood_temperature = route_ood_temperature
        used_idn_temperature = route_idn_temperature
        # Calibrate once after epoch 20, then freeze both temperatures.
        # Re-searching every epoch can create a destructive feedback loop:
        # a sudden temperature jump changes routing, which changes the model
        # and the next epoch's evidence, causing an even larger jump.
        if epoch + 1 == ROUTE_TEMPERATURE_SEARCH_START_EPOCH:
            (
                route_ood_temperature,
                route_idn_temperature,
                next_route_probability,
            ) = search_route_temperatures(
                epoch_ood_evidence, epoch_idn_evidence,
                previous_ood_quantiles, previous_idn_quantiles,
                cfg.openset_ratio, cfg.closeset_noise_ratio
            )
        # evaluate this epoch
        eval_result = evaluate(test_loader, net, device, evaluation)
        test_accuracy = eval_result['accuracy']
        test_loss = eval_result['loss']
        test_auroc = eval_result['auroc']
        test_macro_f1 = eval_result['macro_f1']
        if test_macro_f1 > best_macro_f1:
            best_macro_f1 = test_macro_f1
            best_epoch = epoch + 1
            if cfg.save_model:
                torch.save(net.state_dict(), f'{result_dir}/best_epoch.pth')
                torch.save(net, f'{result_dir}/best_model.pth')

        # logging this epoch
        runtime = time.time() - start_time
        logger.info(
            f'epoch: {epoch + 1} | '
            f'train loss: {train_loss.avg:.4f} | '
            f'train acc: {train_accuracy.avg:.4f} | '
            f'test loss: {test_loss:.4f} | '
            f'test acc: {test_accuracy:.5f} | '
            f'test auroc: {test_auroc:.5f} | '
            f'test macro_f1: {test_macro_f1:.5f} | '
            f'route p(clean/idn/ood): '
            f'{train_clean_prob.avg:.3f}/'
            f'{train_idn_prob.avg:.3f}/'
            f'{train_ood_prob.avg:.3f} | '
            f'ood_tem: {used_ood_temperature:.3f} | '
            f'idn_tem: {used_idn_temperature:.3f} | '
            f'true radius(clean/idn/ood): '
            f'{true_radius_meters[0].avg:.3f}/'
            f'{true_radius_meters[1].avg:.3f}/'
            f'{true_radius_meters[2].avg:.3f} | '
            f'class radius: {net.classfier_head.radius.item():.2f} | '
            f'best f1: {best_macro_f1:.2f}'
        )
        plot_results(result_file=f'{result_dir}/log.txt', layout='2x2')

    wrapup_training(result_dir, best_macro_f1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=RUN_CONFIG['config'])
    parser.add_argument('--synthetic-data', type=str, default=RUN_CONFIG['synthetic_data'])
    parser.add_argument('--noise-type', type=str, default=RUN_CONFIG['noise_type'])
    parser.add_argument('--closeset-noise-ratio', type=float,
                        default=RUN_CONFIG['closeset_noise_ratio'])
    parser.add_argument('--gpu', type=str, default=RUN_CONFIG['gpu'])
    parser.add_argument('--net', type=str, default=RUN_CONFIG['net'])
    parser.add_argument('--batch-size', type=int, default=RUN_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=RUN_CONFIG['lr'])
    parser.add_argument('--lr-decay', type=str, default=RUN_CONFIG['lr_decay'])
    parser.add_argument('--weight-decay', type=float, default=RUN_CONFIG['weight_decay'])
    parser.add_argument('--opt', type=str, default=RUN_CONFIG['opt'])
    parser.add_argument('--warmup-epochs', type=int, default=RUN_CONFIG['warmup_epochs'])
    parser.add_argument('--warmup-lr-scale', type=float, default=RUN_CONFIG['warmup_lr_scale'])
    parser.add_argument('--epochs', type=int, default=RUN_CONFIG['epochs'])
    parser.add_argument('--save-model', action='store_true', default=RUN_CONFIG['save_model'])
    parser.add_argument('--use-fp16', action='store_true', default=RUN_CONFIG['use_fp16'])
    parser.add_argument('--use-grad-accumulate', action='store_true', default=RUN_CONFIG['use_grad_accumulate'])

    parser.add_argument('--project', type=str, default=RUN_CONFIG['project'])
    parser.add_argument('--log', type=str, default=RUN_CONFIG['log'])
    parser.add_argument('--epsilon', type=float, default=RUN_CONFIG['epsilon'])

    parser.add_argument('--eta', type=float, default=RUN_CONFIG['eta'], help='hyper-parameter for balancing target ood probability')
    parser.add_argument('--alpha', type=float, default=RUN_CONFIG['alpha'], help='weight for the entropy loss (clean)')
    parser.add_argument('--beta', type=float, default=RUN_CONFIG['beta'], help='weight for the ood classification loss')
    parser.add_argument('--omega', type=float, default=RUN_CONFIG['omega'], help='weight for the consistency loss')
    parser.add_argument('--decay_epoch', type=int, default=RUN_CONFIG['decay_epoch'], help='weight for the consistency loss')
    parser.add_argument('--seed', type=int, default=RUN_CONFIG['seed'], help='seed')

    parser.add_argument('--rho', type=float, default=RUN_CONFIG['rho'], help='weight for the idn loss')  # TODO: delete

    parser.add_argument('--loss-func-aux', type=str, default=RUN_CONFIG['loss_func_aux'])

    parser.add_argument('--neg-cons', action='store_true', default=RUN_CONFIG['neg_cons'])

    parser.add_argument('--activation', type=str, default=RUN_CONFIG['activation'])
    parser.add_argument('--ablation', action='store_true', default=RUN_CONFIG['ablation'])
    parser.add_argument('--proto-radius', type=float,
                        default=RUN_CONFIG['proto_radius'])
    parser.add_argument('--proto-temperature', type=float,
                        default=RUN_CONFIG['proto_temperature'])
    parser.add_argument('--proto-lambda', type=float,
                        default=RUN_CONFIG['proto_lambda'])
    parser.add_argument('--mutual-lambda', type=float,
                        default=RUN_CONFIG['mutual_lambda'])
    parser.add_argument('--mutual-ramp-epochs', type=int,
                        default=RUN_CONFIG['mutual_ramp_epochs'])
    parser.add_argument('--peer-temperature', type=float,
                        default=RUN_CONFIG['peer_temperature'])

    args = parser.parse_args()

    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])

    assert config.synthetic_data in ['cifar100nc', 'cifar80no','cifar60no']
    assert config.proto_radius > 0.0
    assert config.proto_temperature > 0.0
    assert config.proto_lambda >= 0.0
    assert config.mutual_lambda >= 0.0
    assert config.mutual_ramp_epochs > 0
    assert config.peer_temperature > 0.0
    assert config.noise_type in ['symmetric', 'asymmetric']
    if config.synthetic_data == 'cifar100nc':
        config.openset_ratio = 0.0
    elif config.synthetic_data == 'cifar80no':
        config.openset_ratio = 0.2
    else:
        config.openset_ratio = 0.4
    if config.ablation:
        config.project = f'ablation/{config.project}'
    config.log_freq = LOG_FREQ
    print(config)
    return config


if __name__ == '__main__':
    # Required by Windows when DataLoader starts worker processes with "spawn".
    torch.multiprocessing.freeze_support()
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(
        f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime / 3600:.3f} hours)')
