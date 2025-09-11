import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Teacher 预测熵 H(p) = -sum p log p，数值稳定。
    logits: [B, C]
    return: [B]
    """
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def kl_div_with_temperature(t_logits: torch.Tensor, s_logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    KL(softmax(t/tau) || softmax(s/tau))，逐样本平均。
    t_logits, s_logits: [B, C]
    tau: [B] 或 [1]
    return: [B]
    """
    # broadcast tau -> [B, 1]
    tau = tau.view(-1, 1)
    t_logp = F.log_softmax(t_logits / tau, dim=-1)
    s_logp = F.log_softmax(s_logits / tau, dim=-1)
    t_p = t_logp.exp()
    # KL = sum t_p * (log t_p - log s_p)
    kl = (t_p * (t_logp - s_logp)).sum(dim=-1)
    # 经典 KD 多乘一个 tau^2（可选）；若不想缩放可注释掉
    kl = kl * (tau.squeeze(1) ** 2)
    return kl


def l2_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)


def cosine_similarity(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    计算最后一维上的余弦相似度。
    u, v: [..., D]
    return: [...]
    """
    u_norm = u / (u.norm(dim=-1, keepdim=True) + eps)
    v_norm = v / (v.norm(dim=-1, keepdim=True) + eps)
    return (u_norm * v_norm).sum(dim=-1)


@dataclass
class SANDKDConfig:
    # KD softening
    beta_uncertainty: float = 2.0       # w(x)=exp(-beta * H)
    eta_sigmoid: float = 2.0            # tau(x) 的 Sigmoid 斜率
    tau_min: float = 1.0
    tau_max: float = 4.0

    # geometric soft
    beta_rho: float = 0.5               # 对数范数项权重（相对 geom loss 内部）
    huber_delta: float = 0.2

    # neighbor consistency
    k_neighbors: int = 5
    neighbor_sim_threshold: float = 0.6 # 教师方向的相似性阈值
    # 若设为 True，则 w_ij 使用相似度作为权重；False 则统一权重 1
    neighbor_weight_by_sim: bool = True

    # losses global weights（可在训练中动态调度）
    lambda_sup: float = 1.0
    lambda_kd: float = 1.0
    lambda_geom: float = 0.5
    lambda_nbr: float = 0.25

    # 调度参数（Sigmoid 软启动）：在训练后期逐步开启 geom 与 nbr
    use_sigmoid_schedule: bool = True
    schedule_k_geom: float = 8.0
    schedule_b_geom: float = 0.15        # 在 t/T > b 后快速上升  20 % 处开始起 geom
    schedule_k_nbr: float = 8.0
    schedule_b_nbr: float = 0.15         # 20 % 处开始起 geom

    eps: float = 1e-12                  # 数值稳定


class SANDKDLoss(nn.Module):
    """
    SAND-KD 四个损失项的实现：
      - sup_ce(logits_s, labels)
      - soft_kd(logits_t, logits_s)
      - geom_soft(feat_t, feat_s)
      - neighbor_consistency(feat_t, feat_s)
    其中 feat_* 一般取“最后一层特征”即可。
    """
    def __init__(self, cfg: SANDKDConfig):
        super().__init__()
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss(reduction='none')

    # ------------------------ 1) Supervised CE ------------------------
    def sup_ce(self, logits_s: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits_s: [B, C], labels: [B]
        return: 标量 loss
        """
        ce_per = self.ce(logits_s, labels)  # [B]
        return ce_per.mean()

    # ------------------------ 2) Soft KD ------------------------
    def soft_kd(self, t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
        """
        不确定性权重 + 自适应温度 KD。
        t_logits, s_logits: [B, C]
        """
        cfg = self.cfg
        with torch.no_grad():
            H = entropy_from_logits(t_logits)                     # [B]
            w = torch.exp(-cfg.beta_uncertainty * H)              # [B]
            # 归一化到(0,1]附近，防止过小；也可以不归一化
            w = w / (w.max().detach() + cfg.eps)

            # 自适应温度：tau = tau_min + (tau_max - tau_min)*sigmoid(eta*(H - mean(H)))
            Hc = H - H.mean()
            tau = cfg.tau_min + (cfg.tau_max - cfg.tau_min) * torch.sigmoid(cfg.eta_sigmoid * Hc)

        kl_per = kl_div_with_temperature(t_logits, s_logits, tau) # [B]
        loss = (w * kl_per).mean()
        return loss

    # ------------------------ 3) Geometric soft ------------------------
    def geom_soft(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """
        方向余弦 + 对数范数差的 Huber。
        t_feat, s_feat: [B, D]
        """
        cfg = self.cfg
        # direction
        cos_sim = cosine_similarity(t_feat, s_feat, eps=cfg.eps)  # [B]
        dir_loss = (1.0 - cos_sim).mean()

        # log-norm difference (比例不敏感)
        rho_t = l2_norm(t_feat, eps=cfg.eps)                      # [B,1]
        rho_s = l2_norm(s_feat, eps=cfg.eps)                      # [B,1]
        dlog = (rho_s.log() - rho_t.log()).squeeze(1)             # [B]
        huber = F.huber_loss(dlog, torch.zeros_like(dlog), delta=cfg.huber_delta, reduction='mean')

        return dir_loss + cfg.beta_rho * huber

    # ------------------------ 4) Neighbor consistency ------------------------
    @torch.no_grad()
    def _topk_neighbors_teacher(self, t_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于教师方向的 Top-K 邻域采样，返回 (indices, weights)
          indices: [B, K] 每个样本的 K 个邻居索引（不含自身）
          weights: [B, K] 邻居权重（相似度阈值过滤后）
        """
        cfg = self.cfg
        # 单位方向
        t_u = t_feat / (t_feat.norm(dim=-1, keepdim=True) + cfg.eps)  # [B, D]
        # 余弦相似度矩阵（B,B）
        sim = t_u @ t_u.t()
        B = sim.size(0)

        # 避免选到自己
        sim = sim - torch.eye(B, device=sim.device) * 2.0  # 把对角线设成极小（< threshold）

        # Top-K
        K = min(cfg.k_neighbors, max(B - 1, 1))
        topk_sim, topk_idx = torch.topk(sim, k=K, dim=-1)         # [B, K]

        # 阈值过滤
        mask = (topk_sim >= cfg.neighbor_sim_threshold).float()   # [B, K]
        if cfg.neighbor_weight_by_sim:
            weights = topk_sim.clamp_min(0.0) * mask
        else:
            weights = mask

        # 防止全为 0（回退为均匀 1/K）
        zero_rows = (weights.sum(dim=-1, keepdim=True) <= 0)
        if zero_rows.any():
            weights = torch.where(zero_rows, torch.full_like(weights, 1.0 / K), weights)

        return topk_idx, weights

    def neighbor_consistency(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """
        让教师相似的邻对在学生空间中也相似：E[w_ij * (1 - cos(s_i, s_j))]
        t_feat, s_feat: [B, D]
        """
        cfg = self.cfg
        with torch.no_grad():
            nbr_idx, weights = self._topk_neighbors_teacher(t_feat)  # [B,K], [B,K]

        # 学生方向
        s_u = s_feat / (s_feat.norm(dim=-1, keepdim=True) + cfg.eps)  # [B, D]

        # gather 邻居向量 [B, K, D]
        s_u_i = s_u.unsqueeze(1).expand(-1, nbr_idx.size(1), -1)      # [B,K,D]
        s_u_j = s_u[nbr_idx]                                          # [B,K,D]

        # 余弦相似
        cos_ij = (s_u_i * s_u_j).sum(dim=-1)                          # [B,K]
        loss_ij = (1.0 - cos_ij) * weights                            # [B,K]

        # 归一化（按样本内的权重和）
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)  # [B,1]
        per_sample = (loss_ij / denom).sum(dim=-1)                    # [B]
        return per_sample.mean()

    # ------------------------ 总损失 ------------------------
    def forward(
        self,
        logits_s: torch.Tensor,          # [B, C]
        labels: torch.Tensor,            # [B]
        logits_t: Optional[torch.Tensor] = None,   # [B, C]
        feat_s: Optional[torch.Tensor] = None,     # [B, D]
        feat_t: Optional[torch.Tensor] = None,     # [B, D]
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        返回分项与总损失：
            {
              'loss_total': Tensor[scalar],
              'loss_sup': ...,
              'loss_kd': ...,
              'loss_geom': ...,
              'loss_nbr': ...
            }
        """
        cfg = self.cfg

        # 基础项
        loss_sup = self.sup_ce(logits_s, labels)

        # KD（需要教师 logits）
        loss_kd = torch.tensor(0.0, device=logits_s.device)
        if logits_t is not None:
            loss_kd = self.soft_kd(logits_t, logits_s)

        # geom 与 nbr（需要特征）
        loss_geom = torch.tensor(0.0, device=logits_s.device)
        loss_nbr = torch.tensor(0.0, device=logits_s.device)
        if (feat_s is not None) and (feat_t is not None):
            loss_geom = self.geom_soft(feat_t, feat_s)
            loss_nbr = self.neighbor_consistency(feat_t, feat_s)

        # 调度（Sigmoid 软启动）
        lam_geom = cfg.lambda_geom
        lam_nbr = cfg.lambda_nbr
        if cfg.use_sigmoid_schedule and (step is not None) and (total_steps is not None) and total_steps > 0:
            t = step / float(total_steps)
            s_geom = torch.sigmoid(torch.tensor(cfg.schedule_k_geom * (t - cfg.schedule_b_geom), device=logits_s.device))
            s_nbr = torch.sigmoid(torch.tensor(cfg.schedule_k_nbr * (t - cfg.schedule_b_nbr), device=logits_s.device))
            lam_geom = lam_geom * float(s_geom.item())
            lam_nbr = lam_nbr * float(s_nbr.item())

        loss_total = (
            cfg.lambda_sup * loss_sup
            + cfg.lambda_kd  * loss_kd
            + lam_geom       * loss_geom
            + lam_nbr        * loss_nbr
        )

        return dict(
            loss_total=loss_total,
            loss_sup=loss_sup.detach(),
            loss_kd=loss_kd.detach(),
            loss_geom=loss_geom.detach(),
            loss_nbr=loss_nbr.detach(),
            lam_geom=torch.tensor(lam_geom, device=logits_s.device),
            lam_nbr=torch.tensor(lam_nbr, device=logits_s.device),
        )


# ------------------------ 用法示例 ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, D = 16, 10, 256

    # 假设：来自网络前向
    logits_s = torch.randn(B, C)  # 学生分类输出
    logits_t = torch.randn(B, C)  # 教师分类输出
    feat_s = torch.randn(B, D)    # 学生最后一层特征
    feat_t = torch.randn(B, D)    # 教师最后一层特征
    labels = torch.randint(0, C, (B,))

    cfg = SANDKDConfig()
    criterion = SANDKDLoss(cfg)

    # 训练循环中传入 step/total_steps 以启用软启动
    out = criterion(
        logits_s=logits_s,
        labels=labels,
        logits_t=logits_t,
        feat_s=feat_s,
        feat_t=feat_t,
        step=100, total_steps=1000
    )
    print({k: float(v) if torch.is_tensor(v) else v for k, v in out.items() if 'loss' in k or 'lam_' in k})