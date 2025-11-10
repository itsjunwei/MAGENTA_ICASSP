import numpy as np
import torch
import torch.nn as nn

class MAGENTA_ADPIT(nn.Module):
    """
    MAGENTA wrapped with ADPIT for multi-ACCDOA (Ntracks=3).

    Expects:
      y_pred:  [B, T, 3 * Ntracks * C] = [64, 50, 117]
      y_true:  [B, T, 6, 4, C]     = [64, 50, 6, 4, 13]
    Returns:
      loss (scalar)
    """
    def __init__(self, num_classes: int = 13, ntracks: int = 3, class_counts_path: str = "starss_class_counts.npy",
                 use_class: bool = False, use_miangle: bool = False, use_sat: bool = False, neg_mode: str = "none", eps: float = 1e-8):
        super().__init__()
        self.C      = int(num_classes)
        self.K      = int(ntracks)
        self.eps    = float(eps)

        # MAGENTA toggles
        self.use_class   = bool(use_class)
        self.use_miangle = bool(use_miangle)
        self.use_sat     = bool(use_sat)
        self.neg_mode    = str(neg_mode)

        # ---- rarity priors (mean-1) ----
        counts = np.load(class_counts_path).astype(np.float32)
        IR = float(counts.max() / max(1.0, counts.min()))
        gamma = np.log(IR) / (1.0 + np.log(IR))
        pri = (counts.max() / counts) ** gamma
        pri /= pri.mean()
        inv = 1.0 / (1.0 + pri)

        self.register_buffer('class_weights', torch.from_numpy(pri))   # (C,)
        self.register_buffer('inv_priors',    torch.from_numpy(inv))   # (C,)
        print("Class Priors:\t", self.class_weights)
        print("Inverse Priors:\t", self.inv_priors)

        print(f"MAGENTA | class:{self.use_class} | mia:{self.use_miangle} | sat:{self.use_sat} | neg:{self.neg_mode}")

    # ---------------------------
    # MAGENTA core on one track
    # pred, targ : [B, T, 3, C]  (x,y,z)
    # returns    : [B, T, C]
    # ---------------------------
    def _magenta_tracks_batched(self, pred, targ):
        # pred,targ: [P,B,T,K,3,C]
        xp, yp, zp = pred[..., 0, :], pred[..., 1, :], pred[..., 2, :]
        xt, yt, zt = targ[..., 0, :], targ[..., 1, :], targ[..., 2, :]

        r_true_sq = xt*xt + yt*yt + zt*zt
        r_pred_sq = xp*xp + yp*yp + zp*zp
        r_true = torch.sqrt(r_true_sq + self.eps)
        r_pred = torch.sqrt(r_pred_sq + self.eps)

        # active mask per track
        active  = (r_true > 0.5).to(xp.dtype)
        inactive = 1.0 - active

        a = xp*xt + yp*yt + zp*zt
        cos_theta = torch.clamp(a / (r_pred + self.eps), -1.0, 1.0)

        # 1) under-only radial
        parallel_under_sq = torch.relu(1.0 - a).pow(2)
        if self.use_class:
            cw = self.class_weights.view(1,1,1,1, self.C)
            parallel_under_sq = parallel_under_sq * (1.0 + cw)

        # 2) angular loss
        if self.use_miangle:
            L_ang = 1.0 - cos_theta * cos_theta
        else:
            ex, ey, ez = xt - xp, yt - yp, zt - zp
            err_sq = ex*ex + ey*ey + ez*ez
            e_dot_t = ex*xt + ey*yt + ez*zt
            e_par_sq = (e_dot_t * e_dot_t) / (r_true_sq + self.eps)
            L_ang = torch.clamp(err_sq - e_par_sq, min=0.0)

        # 3) saturation
        if self.use_sat:
            sin2 = 1.0 - cos_theta * cos_theta
            L_sat = torch.relu(r_pred - 1.0).pow(2) * (sin2 + 1.0)
        else:
            L_sat = 0.0

        L_active = parallel_under_sq + L_ang + L_sat

        # Inactive penalty
        base_inactive = r_pred_sq
        if self.neg_mode == "none":
            L_inactive = base_inactive
        elif self.neg_mode == "inv_prior":
            w = self.inv_priors.view(1,1,1,1, self.C)
            L_inactive = w * base_inactive
        else:
            raise ValueError(f"Unknown neg_mode: {self.neg_mode}")

        # combine per track and average across tracks -> [P,B,T,C]
        L = active * L_active + inactive * L_inactive
        return L.mean(dim=3)

    # -----------------------------------
    # Build the 13 ADPIT concatenations
    # target: [B,T,6,4,C]  (dummy_id, [act, x, y, z], class)
    # Returns list of 13 tensors, each [B,T, (3*K), C] (concat tracks along axis=2)
    # -----------------------------------
    @staticmethod
    def _adpit_build_13(target):
        # Multiply (act) * (xyz) for each dummy slot
        # resulting shapes: [B,T,3,C]
        A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]
        B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]
        B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]
        C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]
        C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]
        C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]

        # concat tracks (3 per permutation) -> [B, T, 9, C]
        A0A0A0 = torch.cat((A0, A0, A0), dim=2)

        B0B0B1 = torch.cat((B0, B0, B1), dim=2)
        B0B1B0 = torch.cat((B0, B1, B0), dim=2)
        B0B1B1 = torch.cat((B0, B1, B1), dim=2)
        B1B0B0 = torch.cat((B1, B0, B0), dim=2)
        B1B0B1 = torch.cat((B1, B0, B1), dim=2)
        B1B1B0 = torch.cat((B1, B1, B0), dim=2)

        C0C1C2 = torch.cat((C0, C1, C2), dim=2)
        C0C2C1 = torch.cat((C0, C2, C1), dim=2)
        C1C0C2 = torch.cat((C1, C0, C2), dim=2)
        C1C2C0 = torch.cat((C1, C2, C0), dim=2)
        C2C0C1 = torch.cat((C2, C0, C1), dim=2)
        C2C1C0 = torch.cat((C2, C1, C0), dim=2)

        # paddings like baseline to avoid zeros as targets
        pad4A = B0B0B1 + C0C1C2
        pad4B = A0A0A0 + C0C1C2
        pad4C = A0A0A0 + B0B0B1

        P = [
            A0A0A0 + pad4A,
            B0B0B1 + pad4B, B0B1B0 + pad4B, B0B1B1 + pad4B,
            B1B0B0 + pad4B, B1B0B1 + pad4B, B1B1B0 + pad4B,
            C0C1C2 + pad4C, C0C2C1 + pad4C, C1C0C2 + pad4C,
            C1C2C0 + pad4C, C2C0C1 + pad4C, C2C1C0 + pad4C
        ]  # each [B,T, 3*K, C] with K=3
        targets = torch.stack(P, dim=0) # [P = 13, B, T, 3K, C]
        return targets

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        B, T, D = y_pred.shape
        assert D == 3*self.K*self.C, f"{D} != 3*K*C={3*self.K*self.C}"
        assert y_true.shape[2:] == (6,4,self.C), f"y_true must be [B,T,6,4,C], got {tuple(y_true.shape)}"

        # Build 13 permutations in one tensor
        targets = self._adpit_build_13(y_true)    # [13, B, T, 3K, C]

        # Reshape predictions and targets to [P,B,T,K,3,C]
        pred = y_pred.view(B, T, 3*self.K, self.C).unsqueeze(0) # [1, B, T, 3K, C]
        P = targets.shape[0]
        pred = pred.expand(P, -1, -1, -1, -1)
        pred = pred.view(P, B, T, self.K, 3, self.C)
        targets = targets.view(P, B, T, self.K, 3, self.C)

        # Batched MAGENTA and ADPIT min over permutations
        L_perm = self._magenta_tracks_batched(pred, targets)    # [13, B, T, C]
        L_min, _ = torch.min(L_perm, dim=0) # [B, T, C]
        return L_min.mean()

# ----------------
# Ablation factory
# ----------------
def build_magenta_ablation(name: str, class_counts_path: str = "starss_class_counts.npy", num_classes: int = 13):
    """
    A1: MAGENTA core (under-only radial + geometric perpendicular).
    A2: A1 + rarity prior on radial (use_class=True).
    A3: A2 + inverse-prior inactives (neg_mode='inv_prior').
    A4: A3 + unit-ball saturation.
    """
    n = name.upper().strip()

    # defaults for MAGENTA
    kw = dict(class_counts_path=class_counts_path, num_classes=num_classes, neg_mode="none")

    # --- A1: BASE MAGENTA ---
    if n == "A1":
        pass
    elif n == "M1":
        kw.update(use_miangle=True)

    # --- A2: A1 + Class Priors ---
    elif n == "A2":
        kw.update(use_class=True)
    elif n == "M2":
        kw.update(use_class=True, use_miangle=True)

    # --- A3: A2 + Inverse Priors ---
    elif n == "A3":
        kw.update(use_class=True, neg_mode="inv_prior")
    elif n == "M3":
        kw.update(use_class=True, neg_mode="inv_prior", use_miangle=True)

    # --- A4: A3 + Saturation Loss ---
    elif n == "A4":
        kw.update(use_class=True, neg_mode="inv_prior", use_sat=True)
    elif n == "M4":
        kw.update(use_class=True, neg_mode="inv_prior", use_miangle=True, use_sat=True)

    else:
        raise ValueError(f"Unknown ablation name: {name}")

    return MAGENTA_ADPIT(**kw)