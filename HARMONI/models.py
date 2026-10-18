import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset
from torch_geometric.utils import add_self_loops, coalesce, negative_sampling, remove_self_loops
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-15
MAX_LOGSTD = 10
sc = 0.8


class VNetFeature(torch.nn.Module):
    def __init__(self, input_dim, hidden1, hidden2):
        super(VNetFeature, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class VNetTask(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_families):
        super(VNetTask, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())
            for _ in range(num_families)
        ])
        self.output_dim = output_dim

    def forward(self, x, task_family):
        task_family = task_family.long()
        all_outputs = torch.stack([layer(x) for layer in self.layers], dim=1)
        row_index = torch.arange(x.size(0), device=x.device)
        return all_outputs[row_index, task_family]


class VNet(torch.nn.Module):
    """CMW-Net style class-aware sample weighting module."""

    def __init__(self, input_dim, hidden1, hidden2, output_dim, num_families):
        super(VNet, self).__init__()
        self.feature = VNetFeature(input_dim, hidden1, hidden2)
        # Keep CMW-Net's original attribute spelling for easier comparison.
        self.classfier = VNetTask(hidden2, output_dim, num_families)

    def forward(self, x, num, c=None):
        if c is not None:
            if not torch.is_tensor(c):
                c = torch.tensor(c, dtype=torch.long, device=x.device)
            else:
                c = c.to(device=x.device, dtype=torch.long)
            task_family = c[num.long()]
        else:
            task_family = num.long()
        return self.classfier(self.feature(x), task_family)


class LaplacianGuidedGraphLearner(torch.nn.Module):
    """Feature-driven graph learner with Laplacian-style smoothness support."""

    def __init__(self, input_dim, hidden_dim=128, top_k=8, alpha=0.35, tau=0.2):
        super(LaplacianGuidedGraphLearner, self).__init__()
        self.proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.top_k = top_k
        self.alpha = alpha
        self.tau = tau

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self, x, base_edge_index=None):
        num_nodes = x.size(0)
        k = min(self.top_k, max(num_nodes - 1, 0))

        if k <= 0:
            if base_edge_index is None:
                return x.new_empty((2, 0), dtype=torch.long), x.new_empty((0,))
            return base_edge_index, x.new_ones(base_edge_index.size(1))

        h = F.normalize(self.proj(x), p=2, dim=1)
        sim = torch.matmul(h, h.t()) / max(self.tau, EPS)
        sim.fill_diagonal_(-float('inf'))

        values, cols = torch.topk(sim, k=k, dim=1)
        weights = F.softmax(values, dim=1).reshape(-1)
        rows = torch.arange(num_nodes, device=x.device).view(-1, 1).expand(-1, k).reshape(-1)
        cols = cols.reshape(-1)

        learned_edge_index = torch.stack([rows, cols], dim=0)
        learned_edge_index = torch.cat([learned_edge_index, learned_edge_index.flip(0)], dim=1)
        learned_weight = torch.cat([weights, weights], dim=0) * self.alpha

        if base_edge_index is None:
            return coalesce(learned_edge_index, learned_weight, num_nodes=num_nodes, reduce='sum')

        base_weight = x.new_full((base_edge_index.size(1),), 1.0 - self.alpha)
        edge_index = torch.cat([base_edge_index, learned_edge_index], dim=1)
        edge_weight = torch.cat([base_weight, learned_weight], dim=0)
        return coalesce(edge_index, edge_weight, num_nodes=num_nodes, reduce='sum')


class ModalityFeatureProjector(torch.nn.Module):
    """Project the drug and herb modalities into one common feature space.

    This is the implementation of the feature construction in Eqs. (4)-(5):
    ``X_d_tilde = Norm2(X_d W_d)`` and
    ``X_h_tilde = Norm2(log(1 + X_h) W_h)``.  The two projection matrices are
    deliberately parameters of the VGAE so that they are learned jointly with
    the encoder rather than fitted once during data loading.
    """

    def __init__(self, drug_nodes, drug_dim, herb_dim, output_dim):
        super().__init__()
        if drug_nodes <= 0 or drug_dim <= 0 or herb_dim <= 0 or output_dim <= 0:
            raise ValueError("Modality dimensions and drug_nodes must be positive")
        self.drug_nodes = int(drug_nodes)
        self.drug_dim = int(drug_dim)
        self.herb_dim = int(herb_dim)
        self.output_dim = int(output_dim)
        self.drug_projection = nn.Linear(self.drug_dim, self.output_dim, bias=False)
        self.herb_projection = nn.Linear(self.herb_dim, self.output_dim, bias=False)

    def reset_parameters(self):
        self.drug_projection.reset_parameters()
        self.herb_projection.reset_parameters()

    def forward(self, x):
        expected_dim = self.drug_dim
        if x.dim() != 2 or x.size(1) != expected_dim:
            raise ValueError(
                "Expected raw node features with shape [N, {}], got {}".format(
                    expected_dim, tuple(x.shape)
                )
            )
        if x.size(0) <= self.drug_nodes:
            raise ValueError(
                "Expected {} drug rows followed by at least one herb row, got {} rows".format(
                    self.drug_nodes, x.size(0)
                )
            )

        x_d = x[:self.drug_nodes]
        x_h = x[self.drug_nodes:]
        # Herbal cluster counts are non-negative by definition.  Validate this
        # explicitly so the implementation remains exactly log(1 + X_h).
        if torch.any(x_h < 0):
            raise ValueError("Herb cluster-count features must be non-negative")
        x_h = torch.log1p(x_h)
        x_d = F.normalize(self.drug_projection(x_d), p=2, dim=1)
        x_h = F.normalize(self.herb_projection(x_h), p=2, dim=1)
        return torch.cat([x_d, x_h], dim=0)


def laplacian_smoothness(z, edge_index, edge_weight=None):
    if edge_index.numel() == 0:
        return z.new_tensor(0.0)
    if edge_weight is None:
        edge_weight = z.new_ones(edge_index.size(1))

    src, dst = edge_index
    diff = z[src] - z[dst]
    smoothness = edge_weight * diff.pow(2).sum(dim=1)
    return smoothness.sum() / edge_weight.sum().clamp_min(EPS)


def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


class Encoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = GCNConv(out_channels, out_channels)
        self.scaling_factor = args.scaling_factor

    def forward(self, x, edge_index, edge_weight=None):
        x_ = self.linear1(x)
        x_ = self.propagate(x_, edge_index, edge_weight=edge_weight)

        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1) * self.scaling_factor

        x = self.propagate(x, edge_index, edge_weight=edge_weight)
        return x, x_


class Encoder2(torch.nn.Module):
    """Propensity-oriented branch operating directly on aligned node features.

    The branch implements ``mu_p = c_p Norm2(X W_mu_p)`` and
    ``log(sigma_p) = X W_sigma_p``.  ``X`` is supplied by ``DVGAE.encode2``
    after the shared modality projector, so these weights depend on the
    aligned feature width ``F0`` rather than on the number of nodes ``N``.
    """

    def __init__(self, in_channels, out_channels):
        super(Encoder2, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        x_ = self.linear1(x)

        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1) * sc
        return x, x_


class DVGAE(torch.nn.Module):
    def __init__(self, args, input_dim, N, drug_nodes=None):
        super(DVGAE, self).__init__()
        self.use_lagraph = getattr(args, 'enable_lagraph', False)
        self.lagraph_lambda = float(getattr(args, 'lagraph_lambda', 0.0))
        self.raw_input_dim = int(input_dim)
        self.common_feature_dim = int(getattr(args, 'feature_dim', input_dim))
        self.drug_nodes = int(drug_nodes if drug_nodes is not None else N)
        if self.drug_nodes >= N:
            raise ValueError("drug_nodes must be smaller than the total node count")
        self.feature_projector = ModalityFeatureProjector(
            drug_nodes=self.drug_nodes,
            drug_dim=self.raw_input_dim,
            herb_dim=self.raw_input_dim,
            output_dim=self.common_feature_dim,
        )
        self.encoder1 = Encoder(args, self.common_feature_dim, args.channels)
        self.propensity_dim = int(getattr(args, 'propensity_dim', 2))
        if self.propensity_dim <= 0:
            raise ValueError("propensity_dim must be positive")
        # The propensity branch uses the aligned node features X, whose width
        # is F0, instead of an N-dimensional node-identity lookup table.
        self.encoder2 = Encoder2(self.common_feature_dim, self.propensity_dim)
        if self.use_lagraph:
            self.lagraph_learner = LaplacianGuidedGraphLearner(
                input_dim=self.common_feature_dim,
                hidden_dim=getattr(args, 'lagraph_hidden', args.channels),
                top_k=getattr(args, 'lagraph_topk', 8),
                alpha=getattr(args, 'lagraph_alpha', 0.35),
                tau=getattr(args, 'lagraph_tau', 0.2),
            )
        else:
            self.lagraph_learner = None
        self.decoder = InnerProductDecoder2(self.propensity_dim)
        DVGAE.reset_parameters(self)

    def reset_parameters(self):
        self.feature_projector.reset_parameters()
        reset(self.encoder1)
        reset(self.encoder2)
        reset(self.decoder)

    def encode1(self, *args, **kwargs):
        if args:
            args = (self.feature_projector(args[0]),) + args[1:]
        elif 'x' in kwargs:
            kwargs = dict(kwargs)
            kwargs['x'] = self.feature_projector(kwargs['x'])
        self.__mu1__, self.__logstd1__ = self.encoder1(*args, **kwargs)
        self.__logstd1__ = self.__logstd1__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu1__, self.__logstd1__)
        return z

    def encode2(self, *args, **kwargs):
        if args:
            args = (self.feature_projector(args[0]),) + args[1:]
        elif 'x' in kwargs:
            kwargs = dict(kwargs)
            kwargs['x'] = self.feature_projector(kwargs['x'])
        self.__mu2__, self.__logstd2__ = self.encoder2(*args, **kwargs)
        self.__logstd2__ = self.__logstd2__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu2__, self.__logstd2__)
        return z

    def propagation_graph(self, x, edge_index):
        if self.lagraph_learner is None:
            return edge_index, None
        return self.lagraph_learner(self.feature_projector(x), edge_index)

    def forward(self, x, network_x, edge_index, temp, pos_edge_index, neg_edge_index=None,
                sample_weight=None, reduction='mean', include_kl=True, decoder_training=True,
                prop_edge_weight=None):
        prop_edge_index = edge_index
        if self.lagraph_learner is not None and prop_edge_weight is None:
            prop_edge_index, prop_edge_weight = self.propagation_graph(x, edge_index)
        z1 = self.encode1(x, prop_edge_index, prop_edge_weight)
        z2 = self.encode2(network_x, prop_edge_index)
        sample_loss = self.edge_recon_loss(
            z1, z2, temp, pos_edge_index, neg_edge_index,
            reduction='none',
            decoder_training=decoder_training,
        )
        if include_kl:
            laplacian_loss = None
            if self.lagraph_learner is not None:
                laplacian_loss = laplacian_smoothness(
                    z1, prop_edge_index, prop_edge_weight
                )
            return self.total_loss(
                sample_loss,
                sample_weight=sample_weight,
                laplacian_loss=laplacian_loss,
                lambda_lap=self.lagraph_lambda,
            )

        if sample_weight is not None:
            sample_loss = sample_loss * sample_weight.view_as(sample_loss)
        if reduction == 'none':
            return sample_loss
        if reduction == 'mean':
            return sample_loss.mean()
        if reduction == 'sum':
            return sample_loss.sum()
        raise ValueError("Unsupported reduction: {}".format(reduction))

    def total_loss(self, sample_loss, sample_weight=None, laplacian_loss=None,
                   lambda_lap=0.0, detach_sample_weight=True, include_kl=True):
        """Compute the training objective in Eq. (24).

        ``sample_loss`` contains one reconstruction loss for every edge in
        :math:`B_{tr}`.  Dynamic weights are stop-gradient values for the
        model update.  The bilevel VNet inner step can explicitly disable the
        stop-gradient so that its parameters still receive a meta-gradient.
        """
        if sample_loss.numel() == 0:
            raise ValueError("The training edge batch B_tr must not be empty")

        batch_size = float(sample_loss.numel())
        if sample_weight is None:
            reconstruction = sample_loss.sum() / batch_size
        else:
            sample_weight = sample_weight.view_as(sample_loss)
            if detach_sample_weight:
                sample_weight = sample_weight.detach()
            reconstruction = (sample_weight * sample_loss).sum() / batch_size

        loss = reconstruction
        if include_kl:
            num_nodes = self.__mu1__.size(0)
            loss = loss + (num_nodes / batch_size) * (
                self.kl_loss1() + self.kl_loss2()
            )

        if laplacian_loss is not None and lambda_lap != 0:
            loss = loss + lambda_lap * laplacian_loss
        return loss

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def test(self, z1, z2, temp, pos_edge_index, neg_edge_index, threshold=0.5):

        pos_y = z1.new_ones(pos_edge_index.size(1))
        neg_y = z1.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z1, z2, temp, pos_edge_index, training=False)
        neg_pred = self.decoder(z1, z2, temp, neg_edge_index, training=False)

        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        AUC = roc_auc_score(y, pred)
        AP = average_precision_score(y, pred)

        pred_label = torch.tensor(pred)
        pred_label[pred_label >= threshold] = 1
        pred_label[pred_label < threshold] = 0
        accuracy, sensitivity, precision, specificity, F1_score, mcc = calculate_metrics(y, pred_label.cpu())

        return ['AUC:{:.4f}'.format(AUC), 'AP:{:.4f}'.format(AP),
                'acc:{:.4f}'.format(accuracy.item()), 'sen:{:.4f}'.format(sensitivity.item()),
                'pre:{:.4f}'.format(precision.item()), 'spe:{:.4f}'.format(specificity.item()),
                'f1:{:.4f}'.format(F1_score.item()), 'mcc:{:.4f}'.format(mcc.item())]

    def recon_loss(self, z1, z2, temp, pos_edge_index, neg_edge_index=None):
        return self.edge_recon_loss(z1, z2, temp, pos_edge_index, neg_edge_index, reduction='sum')

    def edge_recon_loss(self, z1, z2, temp, pos_edge_index, neg_edge_index=None,
                        sample_weight=None, reduction='sum', decoder_training=True):

        decode_p = self.decoder(z1, z2, temp, pos_edge_index, training=decoder_training)
        pos_loss = -torch.log(decode_p + EPS)

        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z1.size(0))

        decode_n1 = self.decoder(z1, z2, temp, neg_edge_index, training=decoder_training)
        neg_loss = -torch.log(1 - decode_n1 + EPS)

        loss = torch.cat([pos_loss, neg_loss], dim=0)
        if sample_weight is not None:
            loss = loss * sample_weight.view_as(loss)

        if reduction == 'none':
            return loss
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
        raise ValueError("Unsupported reduction: {}".format(reduction))

    def kl_loss1(self, mu=None, logstd=None):
        mu = self.__mu1__ if mu is None else mu
        logstd = self.__logstd1__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def kl_loss2(self, mu=None, logstd=None):
        mu = self.__mu2__ if mu is None else mu
        logstd = self.__logstd2__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))


class InnerProductDecoder2(torch.nn.Module):
    """Combine similarity and propensity logits for candidate edges.

    The propensity logit follows ``v_p_ij = a_p^T z_i^p + a_p^T z_j^p``.
    ``a_p`` is shared across nodes and optimized jointly with the model, so
    every propensity latent dimension can contribute to prediction.
    """

    def __init__(self, propensity_dim=2):
        super(InnerProductDecoder2, self).__init__()
        if propensity_dim <= 0:
            raise ValueError("propensity_dim must be positive")
        self.propensity_dim = int(propensity_dim)
        self.a_p = nn.Parameter(torch.empty(self.propensity_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.a_p, mean=0.0, std=self.propensity_dim ** -0.5)

    def propensity_logit(self, z2, edge_index):
        if z2.size(-1) != self.propensity_dim:
            raise ValueError(
                "Expected propensity embeddings with width {}, got {}".format(
                    self.propensity_dim, z2.size(-1)
                )
            )
        z2_src = z2[edge_index[0]]
        z2_dst = z2[edge_index[1]]
        return torch.matmul(z2_src, self.a_p) + torch.matmul(z2_dst, self.a_p)

    def forward(self, z1, z2, temp, edge_index, training=True):

        if training:
            z11 = z1.detach().clone()
            vf = (z11[edge_index[0]] * z11[edge_index[1]]).sum(dim=1)
            la = torch.cat((torch.unsqueeze(vf, 1), torch.zeros(torch.unsqueeze(vf, 1).shape, device=z1.device)), 1)
            la_ra = la
            a = F.gumbel_softmax(la_ra, tau=temp, hard=True)[:, :1]
            value_feature = (z1[edge_index[0]] * z1[edge_index[1]]).sum(dim=1)
            value_network = self.propensity_logit(z2, edge_index)
            feature_flag = torch.flatten(a)
            return feature_flag * torch.sigmoid(value_feature) +\
                   (1 - feature_flag) * torch.sigmoid(value_network)

        else:
            value_feature = (z1[edge_index[0]] * z1[edge_index[1]]).sum(dim=1)
            value_network = self.propensity_logit(z2, edge_index)
            return torch.sigmoid(value_feature) * torch.sigmoid(value_feature) +\
                   (1 - torch.sigmoid(value_feature)) * torch.sigmoid(value_network)

    def forward_all(self, z):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj)


# Public model names used by the training entry point.  The four variants
# share the same variational encoder/decoder; the training configuration
# enables the corresponding adaptive-sampling (AS) and relation-modeling
# (RM) components.
class HARMONIAS(DVGAE):
    """HARMONI-AS with adaptive sample weighting."""


class HARMONIRM(DVGAE):
    """HARMONI-RM with Laplacian-regularized relation modeling."""


class HARMONI(DVGAE):
    """HARMONI with adaptive weighting and relation modeling."""


__all__ = [
    'DVGAE', 'HARMONIAS', 'HARMONIRM', 'HARMONI', 'VNet',
    'laplacian_smoothness',
]
