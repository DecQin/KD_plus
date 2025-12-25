import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Dict
from MODELS.reviewkd import hcl

# ---------- Config -----------
@dataclass
class CKDConfig:
    # Dc: int = 2048
    warmup_epochs: int = 24   # 仅 CE+KD 的 epoch 数
    lam_kd: float = 1.2       # 原始logits-KD权重 1.0
    sum_lam: float = 0.4      # 总特征对齐权重 0.5
    lam_mmd: float = 0.04     # MMDz正则权重 0.05


# ---------- Canonical Projections ----------
class CanonicalProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)  # 或 Conv1x1 对CNN
        self.ln = nn.LayerNorm(out_dim)
    def forward(self, h):
        return self.ln(self.proj(h))  # [B, Dc]

def orth_reg(module, lam=1e-4):
    # 近正交正则：||W^T W - I||_F^2
    W = module.proj.weight  # [Dc, Din]
    I = torch.eye(W.shape[0], device=W.device)
    return lam * ((W @ W.t() - I).pow(2).sum())



# ---------- Uncertainty weights from teacher ----------
def teacher_uncertainty_w(z_T, tau=1.0):
    p = (z_T / tau).softmax(-1)
    ent = -(p * (p.clamp_min(1e-12).log())).sum(-1)  # [B]
    w = torch.exp(-ent)  # 熵越大权重越小
    return (w / (w.mean() + 1e-12)).detach()

# ---------- GLS-adaptive Norm–Direction loss in canonical space ----------
class NDLoss(nn.Module):
    def __init__(self, eps=1e-12, huber_delta=0.2):
        super().__init__()
        self.eps = eps
        self.delta = huber_delta
    @staticmethod
    def huber(x, delta):
        a = x.abs()
        return torch.where(a < delta, 0.5*a*a, delta*(a - 0.5*delta))
    def forward(self, hT_c, hS_c, w_samples=None):
        # hT_c, hS_c: [B, Dc] after canonical projection
        eps = self.eps
        # unit vectors
        uT = hT_c / (hT_c.norm(dim=-1, keepdim=True) + eps)
        uS = hS_c / (hS_c.norm(dim=-1, keepdim=True) + eps)
        # direction term
        L_dir_i = 1.0 - (uT * uS).sum(dim=-1)  # [B]
        # radial term (scale-invariant & robust)
        rhoT = hT_c.norm(dim=-1).clamp_min(eps)
        rhoS = hS_c.norm(dim=-1).clamp_min(eps)
        dlog = (rhoS.log() - rhoT.log())
        L_rho_i = self.huber(dlog, self.delta)  # [B]

        # --- GLS weighting (batch anisotropy) ---
        # tangential approx variance via direction residual
        # small-angle: dir ≈ e_t^2 / (2 rho_T^2)
        var_r = L_rho_i.var(unbiased=False).detach() + 1e-12
        var_t = L_dir_i.var(unbiased=False).detach() + 1e-12
        # allocate more weight to higher-variance component (Gauss–Markov)
        beta_rho = (var_t / var_r).clamp(0.1, 10.0).detach()
        # sample uncertainty weight (optional)
        if w_samples is None:
            w = torch.ones_like(L_dir_i)
        else:
            w = w_samples

        loss = (w * (L_dir_i + beta_rho * L_rho_i)).mean()
        stats = dict(beta_rho=float(beta_rho))
        return loss, stats

# ---------- Lightweight MMD (optional) ----------
def mmd_rbf(x, y, sigma=1.0):
    # x,y: [B, Dc], zero-mean before call improves stability
    def k(a, b):
        a2 = (a*a).sum(-1, keepdim=True)
        b2 = (b*b).sum(-1, keepdim=True).t()
        dist = a2 + b2 - 2*a@b.t()
        return torch.exp(-dist / (2*sigma**2))
    Kxx = k(x, x).mean()
    Kyy = k(y, y).mean()
    Kxy = k(x, y).mean()
    return Kxx + Kyy - 2*Kxy

# ---------- Optional Sinkhorn OT for token/pixel matching ----------
def sinkhorn_ot(cost, eps=0.1, iters=50):
    # cost: [N, M], returns transport plan P [N, M]
    N, M = cost.shape
    mu = torch.full((N,), 1.0/N, device=cost.device)
    nu = torch.full((M,), 1.0/M, device=cost.device)
    K = torch.exp(-cost / eps)  # Gibbs kernel
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(iters):
        u = mu / (K @ v + 1e-12)
        v = nu / (K.t() @ u + 1e-12)
    P = torch.diag(u) @ K @ torch.diag(v)
    return P

# ----------- CKDLoss -------------
class CKDLoss(nn.Module):
    def __init__(self, cfg: CKDConfig):
        super().__init__()
        self.cfg=cfg
        self.ce=nn.CrossEntropyLoss()

    def _get_warmup_scale(self, epoch: Optional[int] = None):
        """
        epoch: 从 1 开始的整数，None 表示直接跳过热身（全量 loss）
        返回值 ∈ [0,1]，0 表示热身期，1 表示已完全放开
        """
        if epoch is None:               # 推理或不想用热身
            return 1.0
        # 热身 10 % 总 epoch，可手动调比例
        warmup_epochs = int(self.cfg.warmup_epochs)
        if epoch <= warmup_epochs:
            return 0.0
        return 1.0

    # ------------------------ 1) Supervised CE ------------------------
    def sup_ce(self, logits_s: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits_s: [B, C], labels: [B]
        return: 标量 loss
        """
        ce_per = self.ce(logits_s, labels)  # [B]
        return ce_per.mean()

    # ------------------------ 2) KD ------------------------
    def kd_loss(self, feats_s, feats_t) -> torch.Tensor:
        loss = hcl(feats_s, feats_t)
        return loss.mean()

    # ------------------------ 3) ND ------------------------
    def nd_loss(self, hT_c, hS_c, w_i):
        nd = NDLoss(huber_delta=0.2)
        loss, stats = nd(hT_c,hS_c,w_samples=w_i)
        return loss.mean()

    # ------------------------ 4) MMD ------------------------
    def mmd_loss(self, hT_c, hS_c):
        loss = mmd_rbf(hT_c - hT_c.mean(0, keepdim=True),
                hS_c - hS_c.mean(0, keepdim=True), sigma=1.0)
        return loss.mean()

    # ------------------------ 5) R_orth ----------------------------
    def orth_loss(self, proj_s, proj_t):
        loss = orth_reg(proj_t, 1e-4) + orth_reg(proj_s, 1e-4)
        return loss.mean()

    # -------------------------- 总损失 --------------------------
    def forward(
        self,
        logits_s: torch.Tensor,
        labels: torch.Tensor,
        logits_t: torch.Tensor,
        feat_s: torch.Tensor,
        feat_t: torch.Tensor,
        tz_s: torch.Tensor,
        tz_t: torch.Tensor,
        proj_s: CanonicalProj,
        proj_t: CanonicalProj,
        epoch: Optional[int] = None,   # 新增
    ) -> Dict[str, torch.Tensor]:
        """
        返回分项与总损失：
            {
              'loss_total': Tensor[scalar],
              'loss_sup': ...,
              'loss_kd': ...,
              'loss_nd': ...,
              'loss_mmd': ...,
              'loss_orth': ...,
            }
        """
        # scale = self._get_warmup_scale(epoch)  # 0 或 1
        scale = 1
        cfg = self.cfg
        # CE
        loss_sup = self.ce(logits_s, labels)

        # KD
        loss_kd = self.kd_loss(tz_s, tz_t)

        hT_c = proj_t(feat_t)
        hS_c = proj_s(feat_s)

        w_i = teacher_uncertainty_w(logits_t, tau=1.0)

        # ND
        loss_nd = self.nd_loss(hT_c,hS_c,w_i) * scale

        # MMD
        loss_mmd = self.mmd_loss(hT_c, hS_c) * scale
        
        # orthogonal regularization
        loss_orth = self.orth_loss(proj_t, proj_s)

        # total
        loss_total = (loss_sup
                      + cfg.lam_kd * loss_kd
                      + cfg.sum_lam * loss_nd
                      + cfg.lam_mmd * loss_mmd
                      + loss_orth)

        return dict(
            loss_total=loss_total,
            loss_sup=loss_sup.detach(),
            loss_kd=loss_kd.detach(),
            loss_nd=loss_nd.detach(),
            loss_mmd=loss_mmd.detach(),
            loss_orth=loss_orth.detach(),
            nd_scale=torch.tensor(scale),      # 调试用
        )

# ---------------- quick test ----------------
if __name__ == '__main__':
    torch.manual_seed(0)
    B, C, Dt, Ds = 16, 10, 512, 256

    logits_s = torch.randn(B, C)
    logits_t = torch.randn(B, C)
    feat_s   = torch.randn(B, Ds)
    feat_t   = torch.randn(B, Dt)
    H = W = 32
    tz_s = [torch.randn(B, 256, H, W)]   # 学生侧通道数 = 256
    tz_t = [torch.randn(B,  32, H, W)]   # 老师侧通道数 =  32
    labels   = torch.randint(0, C, (B,))
    proj_s   = CanonicalProj(Ds, 256)
    proj_t   = CanonicalProj(Dt, 256)
    epoch = 20

    cfg = CKDConfig()
    criterion = CKDLoss(cfg)

    out = criterion(logits_s, labels, logits_t, feat_s, feat_t, tz_s, tz_t, proj_s, proj_t)
    print({k: float(v) for k, v in out.items() if 'loss' in k or 'beta' in k})
