import os
from pathlib import Path
import csv
import copy
import json
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.func import functional_call
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, negative_sampling, remove_self_loops
from models import DVGAE, HARMONIAS, HARMONIRM, HARMONI, VNet, laplacian_smoothness

PROJECT_DIR = Path(__file__).resolve().parent

seed = int(os.environ.get("HDI_SEED", "2025"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', choices=['DVGAE', 'HARMONI-AS', 'HARMONI-RM', 'HARMONI'],
    default='DVGAE', help='Model variant to train.'
)
parser.add_argument('--data_dir', type=str, default=str(PROJECT_DIR / 'dataset'))
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--channels', type=int, default=128)
parser.add_argument(
    '--propensity_dim', type=int, default=2,
    help='Propensity-oriented latent dimensionality F_p.',
)
parser.add_argument(
    '--feature_dim', type=int, default=768,
    help='Common node-feature dimension F0 after modality-specific projections.',
)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--vnet_lr', type=float, default=1e-5)
parser.add_argument('--meta_lr', type=float, default=1e-4)
parser.add_argument('--cmw_warmup', type=int, default=500)
parser.add_argument('--cmw_families', type=int, default=3)
parser.add_argument('--vnet_hidden', type=int, default=100)
parser.add_argument('--vnet_loss_clip', type=float, default=3.0)
parser.add_argument('--meta_interval', type=int, default=20)
parser.add_argument(
    '--cmw_alpha', type=float, default=0.2,
    help='CMW mixing coefficient rho in w_e = (1-rho) + rho*q_e.',
)
parser.add_argument(
    '--cmw_weight_min', type=float, default=0.5,
    help='Lower clipping bound for the normalized dynamic weight q_e.',
)
parser.add_argument(
    '--cmw_weight_max', type=float, default=1.5,
    help='Upper clipping bound for the normalized dynamic weight q_e.',
)
parser.add_argument('--print_interval', type=int, default=1)
parser.add_argument('--best_metric', type=str, default='auc_ap', choices=['auc_ap', 'auc', 'ap', 'acc', 'f1', 'mcc'])
parser.add_argument('--eval_threshold', type=float, default=0.5)
parser.add_argument('--tune_threshold', action='store_true')
parser.add_argument('--threshold_metric', type=str, default='mcc', choices=['acc', 'f1', 'mcc', 'youden'])
parser.add_argument('--threshold_steps', type=int, default=181)
parser.add_argument('--no_save_best', action='store_true')
parser.add_argument('--lagraph_hidden', type=int, default=128)
parser.add_argument('--lagraph_topk', type=int, default=8)
parser.add_argument('--lagraph_alpha', type=float, default=0.35)
parser.add_argument('--lagraph_tau', type=float, default=0.2)
parser.add_argument('--lagraph_lambda', type=float, default=0.01)
parser.add_argument('--lagraph_warmup', type=int, default=0)
parser.add_argument('--checkpoint_path', type=str, default='outputs/checkpoint.pt')
parser.add_argument('--vnet_checkpoint_path', type=str, default='outputs/vnet_checkpoint.pt')
parser.add_argument('--metrics_log_path', type=str, default=None)
parser.add_argument('--split_path', type=str, default=None)
parser.add_argument('--prediction_path', type=str, default=None)
parser.add_argument('--validation_prediction_path', type=str, default=None)
parser.add_argument('--result_json_path', type=str, default=None)
parser.add_argument('--inference_warmup', type=int, default=3)
parser.add_argument('--inference_repeats', type=int, default=30)
parser.add_argument('--eval_interval', type=int, default=1)
parser.add_argument(
    '--selection_mode', choices=['validation_best', 'final_epoch'],
    default='validation_best',
    help='Select the validation optimum or use one globally locked final epoch.',
)
parser.add_argument(
    '--validation_only', action='store_true',
    help='Use validation edges as the final evaluation partition and never access test edges.',
)
args = parser.parse_args()
if not 0.0 <= args.cmw_alpha <= 1.0:
    parser.error('--cmw_alpha (rho) must be in [0, 1]')
if args.cmw_weight_min > args.cmw_weight_max:
    parser.error('--cmw_weight_min must not exceed --cmw_weight_max')
args.disable_cmw = args.model in {'DVGAE', 'HARMONI-RM'}
args.enable_lagraph = args.model in {'HARMONI-RM', 'HARMONI'}
Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
Path(args.vnet_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
seed = int(args.seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

MAX_TEMP = 2.0
MIN_TEMP = 0.1
decay_weight = np.log(MAX_TEMP / MIN_TEMP)
decay_step = 150.0
# AUC: 0.9692 AP: 0.9560
# AUC: 0.9822 AP: 0.9773
data_dir = Path(args.data_dir)
drug_feature = pd.read_csv(
    data_dir / "Drug_feature.csv",
    index_col=0,
    engine="python",
).astype(np.float32)  # [1424 rows x 768 columns]
herb_feature = pd.read_csv(
    data_dir / "Herb_Finger_feature.csv",
    index_col=0,
    engine="python",
).astype(np.float32)  # [681 rows x 768 columns]
interaction = pd.read_csv(data_dir / "DHI.csv", engine="python")  # [10033 rows x 4 columns]

drug_features_tensor = torch.tensor(drug_feature.values, dtype=torch.float)
herb_features_tensor = torch.tensor(herb_feature.values, dtype=torch.float)
if drug_features_tensor.shape[1] != herb_features_tensor.shape[1]:
    raise ValueError(
        'Drug and herb feature files must have the same raw width when concatenated; '
        'got {} and {}.'.format(drug_features_tensor.shape[1], herb_features_tensor.shape[1])
    )

drug_index_map = {drug_id: idx for idx, drug_id in enumerate(drug_feature.index)}
herb_index_map = {herb_id: idx for idx, herb_id in enumerate(herb_feature.index)}

edges = []
for _, row in interaction.iterrows():
    drug_id = row['Drug_ID']
    herb_id = row['Food_Herb_ID']

    if drug_id in drug_index_map and herb_id in herb_index_map:
        drug_idx = drug_index_map[drug_id]
        herb_idx = herb_index_map[herb_id] + len(drug_feature)
        edges.append([drug_idx, herb_idx])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
features = torch.cat([drug_features_tensor, herb_features_tensor], dim=0)


def edge_to_tensor(edge_list, add_reverse=True):
    output = []
    for edge in edge_list:
        a, b = edge
        output.append([a, b])
        if add_reverse:
            output.append([b, a])
    return torch.LongTensor(output).T


def split_data(
    node_feature,
    edges,
    A_node_num,
    B_node_num,
    ratio=1,
    train_ratio=0.72,
    meta_ratio=0.08,
    val_ratio=0.10,
    test_ratio=0.10,
):
    if args.split_path:
        split = np.load(args.split_path)

        def global_edges(name):
            values = np.asarray(split[name], dtype=np.int64).copy()
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError('{} in split manifest must have shape [N, 2]'.format(name))
            values[:, 1] += A_node_num
            return values.tolist()

        train_pos_edge = global_edges('train_pos')
        train_neg_edge = global_edges('train_neg')
        meta_pos_edge = global_edges('meta_pos')
        meta_neg_edge = global_edges('meta_neg')
        val_pos_edge = global_edges('val_pos')
        val_neg_edge = global_edges('val_neg')
        test_pos_edge = global_edges('test_pos')
        test_neg_edge = global_edges('test_neg')
        train_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                          pos_edge_label_index=edge_to_tensor(train_pos_edge, add_reverse=False),
                          neg_edge_label_index=edge_to_tensor(train_neg_edge, add_reverse=False))
        meta_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                         pos_edge_label_index=edge_to_tensor(meta_pos_edge, add_reverse=False),
                         neg_edge_label_index=edge_to_tensor(meta_neg_edge, add_reverse=False))
        val_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                        pos_edge_label_index=edge_to_tensor(val_pos_edge, add_reverse=False),
                        neg_edge_label_index=edge_to_tensor(val_neg_edge, add_reverse=False))
        test_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                         pos_edge_label_index=edge_to_tensor(test_pos_edge, add_reverse=False),
                         neg_edge_label_index=edge_to_tensor(test_neg_edge, add_reverse=False))
        return train_data, meta_data, val_data, test_data

    all_possible_edges = set((A, B + A_node_num) for A in range(A_node_num) for B in range(B_node_num))
    positive_edges = set(map(tuple, edges))
    # Stable ordering makes the same seed produce the same ablation split
    # across separate processes and Python hash configurations.
    negative_edges = sorted(all_possible_edges - positive_edges)
    positive_edges = sorted(positive_edges)
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)

    num_edge = len(positive_edges)
    # Cumulative boundaries avoid losing one example to independent flooring.
    # The bundled 7,508 positives become 5,405 / 601 / 751 / 751.
    train_end = int(num_edge * train_ratio)
    meta_end = int(num_edge * (train_ratio + meta_ratio))
    val_end = int(num_edge * (train_ratio + meta_ratio + val_ratio))

    train_pos_edge = positive_edges[:train_end]
    train_neg_edge = negative_edges[:train_end * ratio]

    meta_pos_edge = positive_edges[train_end:meta_end]
    meta_neg_edge = negative_edges[train_end * ratio:meta_end * ratio]

    val_pos_edge = positive_edges[meta_end:val_end]
    val_neg_edge = negative_edges[meta_end * ratio:val_end * ratio]

    test_pos_edge = positive_edges[val_end:]
    test_neg_edge = negative_edges[val_end * ratio:num_edge * ratio]

    train_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                      pos_edge_label_index=edge_to_tensor(train_pos_edge, add_reverse=False),
                      neg_edge_label_index=edge_to_tensor(train_neg_edge, add_reverse=False))
    meta_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                     pos_edge_label_index=edge_to_tensor(meta_pos_edge, add_reverse=False),
                     neg_edge_label_index=edge_to_tensor(meta_neg_edge, add_reverse=False))
    val_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                    pos_edge_label_index=edge_to_tensor(val_pos_edge, add_reverse=False),
                    neg_edge_label_index=edge_to_tensor(val_neg_edge, add_reverse=False))
    test_data = Data(x=node_feature, edge_index=edge_to_tensor(train_pos_edge, add_reverse=True),
                     pos_edge_label_index=edge_to_tensor(test_pos_edge, add_reverse=False),
                     neg_edge_label_index=edge_to_tensor(test_neg_edge, add_reverse=False))
    return train_data, meta_data, val_data, test_data


def build_task_families(edge_index_for_tasks, num_tasks, num_families):
    task_counts = torch.bincount(edge_index_for_tasks[0].cpu(), minlength=num_tasks).float()
    task_count_array = task_counts.view(-1, 1).numpy()
    cluster_num = min(num_families, len(np.unique(task_count_array)))

    if cluster_num <= 1:
        task_families = np.zeros(num_tasks, dtype=np.int64)
        centers = np.array([task_count_array.mean()])
    else:
        kmeans = KMeans(n_clusters=cluster_num, random_state=seed, n_init=10)
        raw_labels = kmeans.fit_predict(task_count_array)
        centers = kmeans.cluster_centers_.reshape(-1)
        sorted_cluster_ids = np.argsort(centers)
        task_families = np.zeros_like(raw_labels, dtype=np.int64)
        for rank, cluster_id in enumerate(sorted_cluster_ids):
            task_families[raw_labels == cluster_id] = rank
        centers = centers[sorted_cluster_ids]

    return torch.LongTensor(task_families), centers.tolist()


def build_edge_task_ids(pos_edge_index, neg_edge_index):
    return torch.cat([pos_edge_index[0], neg_edge_index[0]], dim=0).long()


def vnet_weight_input(sample_loss):
    return sample_loss.detach().clamp(max=args.vnet_loss_clip).view(-1, 1)


def sample_train_negative_edges(pos_edge_index, num_nodes, num_neg_samples=None):
    sampled_from, _ = remove_self_loops(pos_edge_index)
    sampled_from, _ = add_self_loops(sampled_from, num_nodes=num_nodes)
    return negative_sampling(
        sampled_from, num_nodes=num_nodes, num_neg_samples=num_neg_samples
    )


def normalize_cmw_weight(raw_weight):
    normalized = raw_weight.squeeze(1)
    normalized = normalized / normalized.mean().detach().clamp_min(1e-6)
    normalized = normalized.clamp(args.cmw_weight_min, args.cmw_weight_max)
    return (1.0 - args.cmw_alpha) + args.cmw_alpha * normalized


def parse_metric(metric_result, metric_name):
    prefix = metric_name + ':'
    for item in metric_result:
        if item.startswith(prefix):
            return float(item.split(':', 1)[1])
    raise ValueError('Metric {} not found in {}'.format(metric_name, metric_result))


def checkpoint_score(metric_result):
    if args.best_metric == 'auc_ap':
        return parse_metric(metric_result, 'AUC') + parse_metric(metric_result, 'AP')
    return parse_metric(metric_result, args.best_metric)


def binary_metrics_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    sensitivity = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    f1_score = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / (denominator + 1e-10)
    return {
        'acc': accuracy,
        'sen': sensitivity,
        'pre': precision,
        'spe': specificity,
        'f1': f1_score,
        'mcc': mcc,
        'youden': sensitivity + specificity - 1.0,
    }


train_data, meta_data, val_data, test_data = split_data(features, edges, len(drug_feature), len(herb_feature))
train_data, meta_data, val_data, test_data = (
    train_data.to(device), meta_data.to(device), val_data.to(device), test_data.to(device)
)
print('Split sizes: inner-training={} meta-learning={} validation={} test={}'.format(
    train_data.pos_edge_label_index.size(1),
    meta_data.pos_edge_label_index.size(1),
    val_data.pos_edge_label_index.size(1),
    test_data.pos_edge_label_index.size(1),
))

input_dim = train_data.x.shape[1]
N = train_data.x.shape[0]

MODEL_CLASSES = {
    'DVGAE': DVGAE,
    'HARMONI-AS': HARMONIAS,
    'HARMONI-RM': HARMONIRM,
    'HARMONI': HARMONI,
}
model = MODEL_CLASSES[args.model](
    args, input_dim, N, drug_nodes=len(drug_feature)
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# The propensity-oriented branch consumes node features X.  The model applies
# its shared modality projector before the W_mu/W_sigma maps.
propensity_input = train_data.x
l1 = train_data.edge_index
use_cmw = not args.disable_cmw
use_lagraph = args.enable_lagraph
save_best = not args.no_save_best

task_families, task_family_centers = build_task_families(
    train_data.edge_index, N, args.cmw_families
)
task_families = task_families.to(device)
num_task_families = int(task_families.max().item()) + 1

# Initialize VNet for every variant, even when DW is disabled.  Keeping model
# and auxiliary initialization in the same order ensures matched ablation
# partners enter predictor training with the same RNG state.
vnet = VNet(1, args.vnet_hidden, args.vnet_hidden, 1, num_task_families).to(device)
optimizer_vnet = torch.optim.Adam(vnet.parameters(), lr=args.vnet_lr, weight_decay=1e-4)

print('CMW-Net VNet enabled:', use_cmw)
print('LaGraph enabled:', use_lagraph)
print(
    'Ablation design: predictor_positives={} meta_positives={} meta_usage={} '
    'edge_weights={} graph_refinement={}'.format(
        train_data.pos_edge_label_index.size(1),
        meta_data.pos_edge_label_index.size(1),
        'VNet-only' if use_cmw else 'none',
        'learned' if use_cmw else 'w_e=1',
        'on' if use_lagraph else 'off',
    )
)
if use_lagraph:
    print('LaGraph config: topk={} hidden={} alpha={} tau={} lambda={} warmup={}'.format(
        args.lagraph_topk, args.lagraph_hidden, args.lagraph_alpha, args.lagraph_tau,
        args.lagraph_lambda, args.lagraph_warmup
    ))
print('Task family centers:', ['{:.2f}'.format(center) for center in task_family_centers])
cmw_output_min = (1.0 - args.cmw_alpha) + args.cmw_alpha * args.cmw_weight_min
cmw_output_max = (1.0 - args.cmw_alpha) + args.cmw_alpha * args.cmw_weight_max
print(
    'CMW config: warmup={} rho={} q_clip=[{},{}] w_range=[{:.3f},{:.3f}] '
    'meta_lr={} vnet_lr={} interval={} loss_clip={}'.format(
        args.cmw_warmup, args.cmw_alpha,
        args.cmw_weight_min, args.cmw_weight_max,
        cmw_output_min, cmw_output_max,
        args.meta_lr, args.vnet_lr, args.meta_interval, args.vnet_loss_clip,
    )
)

def propagation_graph(edge_index_for_propagation, epoch=None):
    if not use_lagraph:
        return edge_index_for_propagation, None
    if epoch is not None and epoch <= args.lagraph_warmup:
        return edge_index_for_propagation, None
    return model.propagation_graph(train_data.x, edge_index_for_propagation)


def lagraph_loss(z, edge_index_for_propagation, edge_weight):
    if not use_lagraph:
        return z.new_tensor(0.0)
    return laplacian_smoothness(z, edge_index_for_propagation, edge_weight)


def update_vnet(temp, epoch):
    model.train()
    vnet.train()
    neg_edge_index = sample_train_negative_edges(train_data.pos_edge_label_index, N)
    edge_task_ids = build_edge_task_ids(train_data.pos_edge_label_index, neg_edge_index).to(device)

    prop_edge_index, prop_edge_weight = propagation_graph(l1, epoch)
    z1 = model.encode1(train_data.x, prop_edge_index, prop_edge_weight)
    z2 = model.encode2(propensity_input, prop_edge_index)
    sample_loss = model.edge_recon_loss(
        z1, z2, temp, train_data.pos_edge_label_index,
        neg_edge_index, reduction='none'
    )
    raw_weight = vnet(vnet_weight_input(sample_loss), edge_task_ids, task_families)
    sample_weight = normalize_cmw_weight(raw_weight)
    laplacian_loss = None
    if use_lagraph and epoch > args.lagraph_warmup:
        laplacian_loss = lagraph_loss(z1, prop_edge_index, prop_edge_weight)
    inner_loss = model.total_loss(
        sample_loss,
        sample_weight=sample_weight,
        laplacian_loss=laplacian_loss,
        lambda_lap=args.lagraph_lambda,
        # The VNet inner problem needs gradients through w_e(phi).  The
        # stop-gradient from Eq. (24) is applied in the actual model update.
        detach_sample_weight=False,
    )

    named_params = dict(model.named_parameters())
    grads = torch.autograd.grad(
        inner_loss, tuple(named_params.values()),
        create_graph=True, allow_unused=True
    )
    updated_params = {
        name: param if grad is None else param - args.meta_lr * grad
        for (name, param), grad in zip(named_params.items(), grads)
    }

    was_training = model.training
    model.eval()
    meta_loss = functional_call(
        model, updated_params,
        (
            meta_data.x, meta_data.x, meta_data.edge_index, temp,
            meta_data.pos_edge_label_index, meta_data.neg_edge_label_index,
            None, 'mean', False, False
        )
    )
    if was_training:
        model.train()

    optimizer.zero_grad()
    optimizer_vnet.zero_grad()
    meta_loss.backward()
    optimizer_vnet.step()
    optimizer.zero_grad()
    return meta_loss.detach().item()


def update_vnet_without_advancing_predictor_rng(temp, epoch):
    """Update only VNet while preserving the predictor's random stream.

    The meta step performs its own negative sampling, variational sampling and
    Gumbel sampling.  Restoring all RNG states afterwards ensures that a DW
    run uses exactly the same predictor-step randomness as its non-DW partner
    with the same seed.
    """
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    cuda_devices = []
    if device.type == 'cuda':
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            return update_vnet(temp, epoch)
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)


def train(epoch):
    temp = np.maximum(MAX_TEMP * np.exp(-(epoch - 1) / decay_step * decay_weight), MIN_TEMP)

    model.train()
    meta_loss = None
    if use_cmw and epoch > args.cmw_warmup and args.meta_interval > 0 and epoch % args.meta_interval == 0:
        meta_loss = update_vnet_without_advancing_predictor_rng(temp, epoch)

    optimizer.zero_grad()
    neg_edge_index = sample_train_negative_edges(train_data.pos_edge_label_index, N)
    edge_task_ids = build_edge_task_ids(train_data.pos_edge_label_index, neg_edge_index).to(device)
    prop_edge_index, prop_edge_weight = propagation_graph(l1, epoch)
    z1 = model.encode1(train_data.x, prop_edge_index, prop_edge_weight)
    z2 = model.encode2(propensity_input, prop_edge_index)

    sample_loss = model.edge_recon_loss(
        z1, z2, temp, train_data.pos_edge_label_index,
        neg_edge_index, reduction='none'
    )
    cmw_weight_mean = None
    if use_cmw and epoch > args.cmw_warmup:
        with torch.no_grad():
            raw_weight = vnet(vnet_weight_input(sample_loss), edge_task_ids, task_families)
            sample_weight = normalize_cmw_weight(raw_weight)
        cmw_weight_mean = sample_weight.mean().item()
        # total_loss applies sg[w_e(phi)] for the model update.
    else:
        sample_weight = None
    laplacian_loss = None
    if use_lagraph and epoch > args.lagraph_warmup:
        laplacian_loss = lagraph_loss(z1, prop_edge_index, prop_edge_weight)
    loss = model.total_loss(
        sample_loss,
        sample_weight=sample_weight,
        laplacian_loss=laplacian_loss,
        lambda_lap=args.lagraph_lambda,
    )
    lagraph_reg = (
        z1.new_tensor(0.0) if laplacian_loss is None
        else args.lagraph_lambda * laplacian_loss
    )
    loss.backward()
    optimizer.step()
    return loss, meta_loss, cmw_weight_mean, lagraph_reg.detach().item()


def edge_prediction_scores(pos_edge_index, neg_edge_index, selected_list, epoch=None):
    model.eval()
    with torch.no_grad():
        prop_edge_index, prop_edge_weight = propagation_graph(selected_list, epoch)
        z1 = model.encode1(train_data.x, prop_edge_index, prop_edge_weight)
        z2 = model.encode2(propensity_input, prop_edge_index)
        pos_y = z1.new_ones(pos_edge_index.size(1))
        neg_y = z1.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = model.decoder(z1, z2, 1.0, pos_edge_index, training=False)
        neg_pred = model.decoder(z1, z2, 1.0, neg_edge_index, training=False)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
    return y.detach().cpu().numpy(), pred.detach().cpu().numpy()


def tune_threshold(pos_edge_index, neg_edge_index, selected_list, epoch=None):
    y_true, y_score = edge_prediction_scores(pos_edge_index, neg_edge_index, selected_list, epoch)
    best_threshold = args.eval_threshold
    best_score = -float('inf')
    steps = max(args.threshold_steps, 2)

    for threshold in np.linspace(0.05, 0.95, steps):
        metrics = binary_metrics_at_threshold(y_true, y_score, threshold)
        score = metrics[args.threshold_metric]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def ceshi(pos_edge_index, neg_edge_index, selected_list, threshold=None, epoch=None):
    threshold = args.eval_threshold if threshold is None else threshold
    model.eval()
    with torch.no_grad():
        prop_edge_index, prop_edge_weight = propagation_graph(selected_list, epoch)
        z1 = model.encode1(train_data.x, prop_edge_index, prop_edge_weight)
        z2 = model.encode2(propensity_input, prop_edge_index)
    result = model.test(z1, z2, 1.0, pos_edge_index, neg_edge_index, threshold=threshold)
    return result


best_score = -float('inf')
best_epoch = 0
best_result = None
best_model_state = None
best_vnet_state = None
metrics_log_file = None
metrics_log_writer = None

if args.metrics_log_path:
    metrics_log_dir = os.path.dirname(args.metrics_log_path)
    if metrics_log_dir:
        os.makedirs(metrics_log_dir, exist_ok=True)
    metrics_log_file = open(args.metrics_log_path, 'w', newline='', encoding='utf-8')
    metrics_log_writer = csv.DictWriter(
        metrics_log_file,
        fieldnames=[
            'epoch', 'loss', 'cmw_weight_mean', 'meta_loss', 'lagraph_reg',
            'best_epoch', 'best_score',
            'AUC', 'AP', 'acc', 'sen', 'pre', 'spe', 'f1', 'mcc'
        ]
    )
    metrics_log_writer.writeheader()

for epoch in range(1, args.epochs + 1):
    loss, meta_loss, cmw_weight_mean, lagraph_reg = train(epoch)
    loss_value = loss.item()

    should_evaluate = (
        epoch == 1
        or epoch == args.epochs
        or epoch % max(args.eval_interval, 1) == 0
    )
    is_best = False
    if should_evaluate:
        val_pos, val_neg = val_data.pos_edge_label_index, val_data.neg_edge_label_index
        with torch.no_grad():
            result = ceshi(val_pos, val_neg, val_data.edge_index, epoch=epoch)
            val_score = checkpoint_score(result)
            should_select = (
                val_score > best_score
                if args.selection_mode == 'validation_best'
                else epoch == args.epochs
            )
            if save_best and should_select:
                best_score = val_score
                best_epoch = epoch
                best_result = result
                best_model_state = copy.deepcopy(model.state_dict())
                best_vnet_state = copy.deepcopy(vnet.state_dict()) if use_cmw else None
                is_best = True
            if metrics_log_writer is not None:
                metrics_log_writer.writerow({
                    'epoch': epoch,
                    'loss': '{:.6f}'.format(loss_value),
                    'cmw_weight_mean': '' if cmw_weight_mean is None else '{:.6f}'.format(cmw_weight_mean),
                    'meta_loss': '' if meta_loss is None else '{:.6f}'.format(meta_loss),
                    'lagraph_reg': '{:.6f}'.format(lagraph_reg),
                    'best_epoch': best_epoch,
                    'best_score': '{:.6f}'.format(best_score),
                    'AUC': parse_metric(result, 'AUC'),
                    'AP': parse_metric(result, 'AP'),
                    'acc': parse_metric(result, 'acc'),
                    'sen': parse_metric(result, 'sen'),
                    'pre': parse_metric(result, 'pre'),
                    'spe': parse_metric(result, 'spe'),
                    'f1': parse_metric(result, 'f1'),
                    'mcc': parse_metric(result, 'mcc'),
                })
                metrics_log_file.flush()
    cmw_info = 'cmw_w:{:.4f}'.format(cmw_weight_mean) if cmw_weight_mean is not None else 'cmw_w:-'
    meta_info = 'meta:{:.4f}'.format(meta_loss) if meta_loss is not None else 'meta:-'
    lagraph_info = 'lagraph:{:.4f}'.format(lagraph_reg) if use_lagraph else 'lagraph:off'
    should_print = should_evaluate and (
        (args.print_interval > 0 and epoch % args.print_interval == 0)
        or epoch == 1 or epoch == args.epochs or is_best
    )
    if should_print:
        best_info = 'best:{}:{:.4f}'.format(best_epoch, best_score) if save_best else 'best:off'
        log_items = [
            epoch,
            'loss:{:.4f}'.format(loss_value),
            cmw_info,
            meta_info,
            lagraph_info,
        ]
        log_items.extend([best_info, result])
        print(*log_items)

if metrics_log_file is not None:
    metrics_log_file.close()

if save_best and best_model_state is not None:
    model.load_state_dict(best_model_state)
    if use_cmw and best_vnet_state is not None:
        vnet.load_state_dict(best_vnet_state)
    print('Loaded best validation checkpoint from epoch {} {}'.format(best_epoch, best_result))

torch.save(model.state_dict(), args.checkpoint_path)
if use_cmw:
    torch.save(vnet.state_dict(), args.vnet_checkpoint_path)
model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

final_threshold = args.eval_threshold
if args.tune_threshold:
    final_threshold, threshold_score = tune_threshold(
        val_data.pos_edge_label_index, val_data.neg_edge_label_index, val_data.edge_index,
        epoch=args.epochs
    )
    print('Tuned eval threshold: {:.4f} validation_{}:{:.4f}'.format(
        final_threshold, args.threshold_metric, threshold_score
    ))

if args.validation_only:
    test_pos, test_neg = val_data.pos_edge_label_index, val_data.neg_edge_label_index
else:
    test_pos, test_neg = test_data.pos_edge_label_index, test_data.neg_edge_label_index
result = ceshi(test_pos, test_neg, test_data.edge_index, threshold=final_threshold, epoch=args.epochs)
print(result)

val_pos, val_neg = val_data.pos_edge_label_index, val_data.neg_edge_label_index
val_y, val_score = edge_prediction_scores(
    val_pos, val_neg, val_data.edge_index, epoch=args.epochs
)
val_metrics = binary_metrics_at_threshold(val_y, val_score, final_threshold)
val_metrics['AUC'] = float(roc_auc_score(val_y, val_score))
val_metrics['AP'] = float(average_precision_score(val_y, val_score))

test_y, test_score = edge_prediction_scores(
    test_pos, test_neg, test_data.edge_index, epoch=args.epochs
)
test_metrics = binary_metrics_at_threshold(test_y, test_score, final_threshold)
test_metrics['AUC'] = float(roc_auc_score(test_y, test_score))
test_metrics['AP'] = float(average_precision_score(test_y, test_score))

for _ in range(max(args.inference_warmup, 0)):
    edge_prediction_scores(test_pos, test_neg, test_data.edge_index, epoch=args.epochs)
if device.type == 'cuda':
    torch.cuda.synchronize()
inference_times = []
for _ in range(max(args.inference_repeats, 1)):
    started = time.perf_counter()
    edge_prediction_scores(test_pos, test_neg, test_data.edge_index, epoch=args.epochs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_times.append((time.perf_counter() - started) * 1000.0)

model_parameters = sum(parameter.numel() for parameter in model.parameters())
aux_parameters = sum(parameter.numel() for parameter in vnet.parameters()) if use_cmw else 0
model_size_bytes = os.path.getsize(args.checkpoint_path)
if use_cmw and os.path.exists(args.vnet_checkpoint_path):
    model_size_bytes += os.path.getsize(args.vnet_checkpoint_path)

if args.prediction_path:
    prediction_dir = os.path.dirname(args.prediction_path)
    if prediction_dir:
        os.makedirs(prediction_dir, exist_ok=True)
    test_edges = torch.cat([test_pos, test_neg], dim=1).detach().cpu().numpy().T
    prediction = pd.DataFrame({
        'seed': seed,
        'evaluation_split': 'validation' if args.validation_only else 'test',
        'selection_mode': args.selection_mode,
        'drug_index': test_edges[:, 0],
        'herb_index': test_edges[:, 1] - len(drug_feature),
        'label': test_y.astype(np.int64),
        'score': test_score,
    })
    prediction.to_csv(args.prediction_path, index=False, encoding='utf-8-sig')

if args.validation_prediction_path:
    prediction_dir = os.path.dirname(args.validation_prediction_path)
    if prediction_dir:
        os.makedirs(prediction_dir, exist_ok=True)
    val_edges = torch.cat([val_pos, val_neg], dim=1).detach().cpu().numpy().T
    prediction = pd.DataFrame({
        'seed': seed,
        'drug_index': val_edges[:, 0],
        'herb_index': val_edges[:, 1] - len(drug_feature),
        'label': val_y.astype(np.int64),
        'score': val_score,
    })
    prediction.to_csv(args.validation_prediction_path, index=False, encoding='utf-8-sig')

if args.result_json_path:
    result_dir = os.path.dirname(args.result_json_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    payload = dict(test_metrics)
    payload.update({
        'validation_' + key: value
        for key, value in val_metrics.items()
    })
    payload.update({
        'seed': seed,
        'epochs': args.epochs,
        'eval_interval': max(args.eval_interval, 1),
        'best_epoch': best_epoch,
        'threshold': final_threshold,
        'best_val_auc_ap': best_score,
        'parameter_count': model_parameters + aux_parameters,
        'inference_parameter_count': model_parameters,
        'training_aux_parameter_count': aux_parameters,
        'model_size_mb': model_size_bytes / (1024.0 ** 2),
        'inference_ms_mean': float(np.mean(inference_times)),
        'inference_ms_std': float(np.std(inference_times, ddof=1)) if len(inference_times) > 1 else 0.0,
        'inference_repeats': len(inference_times),
        'device': str(device),
        'split_path': args.split_path,
        'split_ratios': {
            'inner_training': 0.72,
            'meta_learning': 0.08,
            'validation': 0.10,
            'test': 0.10,
        },
        'split_counts': {
            'inner_training': int(train_data.pos_edge_label_index.size(1)),
            'meta_learning': int(meta_data.pos_edge_label_index.size(1)),
            'validation': int(val_data.pos_edge_label_index.size(1)),
            'test': int(test_data.pos_edge_label_index.size(1)),
        },
    })
    with open(args.result_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
