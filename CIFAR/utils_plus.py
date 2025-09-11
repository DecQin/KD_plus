import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# h_T, h_S: [B, D] teacher/student features (after projection if needed)
# z_T, z_S: [B, C] logits
def kd_loss(z_T, z_S, tau=4.0):
    t = (z_T / tau).log_softmax(-1)
    s = (z_S / tau).log_softmax(-1)
    return (t.exp() * (t - s)).sum(dim=-1).mean() * (tau**2)

def norm_dir_loss(h_T, h_S, beta_rho=0.5, eps=1e-12):
    # direction
    u_T = h_T / (h_T.norm(dim=-1, keepdim=True) + eps)
    u_S = h_S / (h_S.norm(dim=-1, keepdim=True) + eps)
    L_dir = (1 - (u_T * u_S).sum(dim=-1)).mean()
    # norm  (replace by Huber(log) if needed)
    rho_T = h_T.norm(dim=-1)
    rho_S = h_S.norm(dim=-1)
    L_rho = ((rho_S - rho_T) ** 2).mean()
    return L_dir + beta_rho * L_rho

# 使用时：total = ce + λ_KD*kd_loss + Σ_l α_l*(λ_u*L_dir + λ_ρ*L_rho)；
# 若不稳定，把 ((rho_S - rho_T)**2) 换成 Huber(log(rho_S)-log(rho_T))。

@dataclass
class SimKDConfig:
    lambda_kd: float = 1.0       # KD损失权重
    alpha_l: float = 0.5         # 特征一致性损失的总权重（对应原公式的alpha_l）
    huber_delta: float = 0.2     # 0.2

class SimKDLoss(nn.Module):
    """
    Sim-KD 三个损失项的实现：
      - sup_ce(logits_s, labels)
      - kd_loss(logits_t, logits_s, tau=4.0):
      - norm_dir_loss(feat_t, feat_s, beta_rho=0.5, eps=1e-12)
    其中 feat_* 一般取“最后一层特征”即可。
    """
    def __init__(self, cfg: SimKDConfig):
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

    # ------------------------ 2) KD ------------------------
    def kd_loss(self, t_logits: torch.Tensor, s_logits: torch.Tensor, tau=4.0) -> torch.Tensor:
        t = (t_logits / tau).log_softmax(-1)
        s = (s_logits / tau).log_softmax(-1)
        loss = (t.exp() * (t - s)).sum(dim=-1).mean() * (tau**2)
        return loss.mean()

    # ------------------------ 3) Norm Dir ------------------------
    def norm_dir_loss(self, t_feat: torch.Tensor, s_feat: torch.Tensor, beta_rho=0.5, eps=1e-12) -> torch.tensor: # detla 可选 
        # direction
        u_T = t_feat / (t_feat.norm(dim=-1, keepdim=True) + eps)
        u_S = s_feat / (s_feat.norm(dim=-1, keepdim=True) + eps)
        L_dir = (1 - (u_T * u_S).sum(dim=-1)).mean()
        # norm  (replace by Huber(log) if needed)
        rho_T = t_feat.norm(dim=-1)
        rho_S = s_feat.norm(dim=-1)
        L_rho = ((rho_S - rho_T) ** 2).mean()
        # Huber损失替换 (可选)
        # dlog = torch.log(rho_S)-torch.log(rho_T)        # [B]
        # L_rho = F.huber_loss(dlog, torch.zeros_like(dlog))
        # L_rho = F.huber_loss(rho_S - rho_T)  # Huber损失对异常值更稳健
        loss = L_dir + beta_rho * L_rho
        return loss.mean()

    # ------------------------ 总损失 ------------------------
    def forward(
        self,
        logits_s: torch.Tensor,          # [B, C]
        labels: torch.Tensor,            # [B]
        logits_t: Optional[torch.Tensor] = None,   # [B, C]
        feat_s: Optional[torch.Tensor] = None,     # [B, D]
        feat_t: Optional[torch.Tensor] = None,     # [B, D]
        # step: Optional[int] = None,
        # total_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        返回分项与总损失：
            {
              'loss_total': Tensor[scalar],
              'loss_sup': ...,
              'loss_kd': ...,
              'loss_nd': ...,
            }
        """
        cfg = self.cfg

        # 基础项
        loss_sup = self.sup_ce(logits_s, labels)

        # KD（需要教师 logits）
        # loss_kd = torch.tensor(0.0, device=logits_s.device)
        # if logits_t is not None:
        loss_kd = self.kd_loss(logits_t, logits_s)
        
        # ND (需要特征)
        loss_nd = self.norm_dir_loss(feat_t, feat_s)

        loss_total = (
            loss_sup
            + cfg.lambda_kd * loss_kd
            + cfg.alpha_l * loss_nd
        )

        return dict(
            loss_total=loss_total,
            loss_sup=loss_sup.detach(),
            loss_kd=loss_kd.detach(),
            loss_nd=loss_nd.detach(),
            # lam_geom=torch.tensor(lam_geom, device=logits_s.device),
            # lam_nbr=torch.tensor(lam_nbr, device=logits_s.device),
        )

if __name__=='__main__':
    torch.manual_seed(0)
    B, C, D = 16, 10, 256

    # 假设：来自网络前向
    logits_s = torch.randn(B, C)  # 学生分类输出
    logits_t = torch.randn(B, C)  # 教师分类输出
    feat_s = torch.randn(B, D)    # 学生最后一层特征
    feat_t = torch.randn(B, D)    # 教师最后一层特征
    labels = torch.randint(0, C, (B,))

    cfg = SimKDConfig()
    criterion = SimKDLoss(cfg)

    # 训练循环中传入 step/total_steps 以启用软启动
    out = criterion(
        logits_s=logits_s,
        labels=labels,
        logits_t=logits_t,
        feat_s=feat_s,
        feat_t=feat_t
    )
    print({k: float(v) if torch.is_tensor(v) else v for k, v in out.items() if 'loss' in k or 'lam_' in k})
