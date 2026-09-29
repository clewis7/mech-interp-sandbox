import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LeakyRNN
from generate_data import generate_task_data
from utils import PCAProjector, trajectory_colors

IGNORE_INDEX = -100


class Session:
    """Everything a task switch has to rebuild.

    Parameters
    ----------
    task : str
        Mod-Cog task name.
    n_trials : int
        Size of the generated trial bank.
    n_probe : int
        Held-out trials kept fixed for the trajectory view.
    tau_ms : float
        Membrane time constant; alpha = dt / tau.
    resp_weight : float
        Loss weight on response steps; fixation dominates the labels otherwise.
    """

    def __init__(
        self,
        task,
        device="cuda",
        seed=0,
        n_trials=8192,
        n_probe=64,
        val_frac=0.1,
        n_hidden=64,
        tau_ms=200,
        sigma_rec=0.05,
        activation="softplus",
        rec_gain=0.9,
        l2_rate=1e-2,
        l2_weight=1e-2,
        batch=256,
        lr=1e-3,
        grad_clip=1.0,
        resp_weight=5.0,
        traj_scale=1.0,
    ):
        torch.manual_seed(seed)
        self.task = task
        self.device = device
        self.batch = batch
        self.grad_clip = grad_clip
        self.resp_weight = resp_weight
        self.l2_rate = l2_rate
        self.l2_weight = l2_weight
        self.n_hidden = n_hidden
        self.sigma_rec = sigma_rec
        self.activation = activation
        self.rec_gain = rec_gain

        self.step = 0
        self.loss = torch.zeros((), device=device)

        self.acc = 0.0
        self.done = False
        self._hits = 0

        data = generate_task_data(task, n_trials, device=device)
        self.obs = data["obs"]
        self.labels = data["labels"]
        self.lengths = data["lengths"]
        self.n_ring = data["n_ring"]
        self.n_out = data["n_out"]
        self.dt = data["dt"]

        n_total, self.t_max, self.n_in = self.obs.shape
        self.alpha = self.dt / tau_ms

        perm = torch.randperm(n_total, device=device)
        n_val = int(val_frac * n_total)
        self.val_idx, self.train_idx = perm[:n_val], perm[n_val:]
        self.probe_idx = self.val_idx[:n_probe]

        self.model = LeakyRNN(
            self.n_in,
            n_hidden,
            self.n_out,
            self.alpha,
            sigma_rec=sigma_rec,
            activation=activation,
            rec_gain=rec_gain,
        ).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        probe_x, probe_y, probe_lens, _ = self.make_batch(self.probe_idx)
        self.projector = PCAProjector(self.model, probe_x, probe_lens, scale=traj_scale)
        self.projector.update(refit=True)

        self.init_positions = self.projector.positions.cpu().numpy()
        self.colors = trajectory_colors(
            probe_y.cpu().numpy(), probe_lens.cpu().numpy(), self.n_ring
        )

    def make_batch(self, idx):
        """Gather trials and zero everything past the end of each one."""
        x = self.obs[idx]
        lens = self.lengths[idx]
        valid = torch.arange(x.shape[1], device=self.device)[None] < lens[:, None]
        return x * valid[..., None], self.labels[idx], lens, valid

    def compute_loss(self, logits, y, hidden, mask):
        """Weighted cross-entropy plus the model's rate and weight penalties."""
        ce = F.cross_entropy(
            logits.reshape(-1, self.n_out),
            y.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(y)
        w = torch.where(y > 0, self.resp_weight, 1.0) * (y != IGNORE_INDEX)
        task_loss = (ce * w).sum() / w.sum()
        reg = self.model.regularization(
            hidden, mask, rate_weight=self.l2_rate, weight_weight=self.l2_weight
        )
        return task_loss + reg, task_loss.detach()

    def train_step(self):
        """One optimizer step. Returns the task loss, detached."""
        idx = self.train_idx[
            torch.randint(len(self.train_idx), (self.batch,), device=self.device)
        ]
        x, y, lens, mask = self.make_batch(idx)

        logits, hidden = self.model(x)
        loss, task_loss = self.compute_loss(logits, y, hidden, mask)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()

        self.step += 1
        self.loss = task_loss
        return task_loss

    @torch.no_grad()
    def accuracy(self, n=2048):
        """Trial accuracy on a random validation sample: fixation held on every
        fixation step AND the last step correct.
        """
        was_training = self.model.training
        self.model.eval()
        idx = self.val_idx[torch.randint(len(self.val_idx), (n,), device=self.device)]
        x, y, lens, _ = self.make_batch(idx)
        logits, _ = self.model(x)
        pred = logits.argmax(-1)
        fix_ok = ~((pred != 0) & (y == 0)).any(dim=1)
        rows = torch.arange(n, device=self.device)
        resp_ok = pred[rows, lens - 1] == y[rows, lens - 1]
        self.model.train(was_training)
        return (fix_ok & resp_ok).float().mean().item()

    def check_stop(self, target_acc=0.98, patience=3, n=2048):
        self.acc = self.accuracy(n)
        self._hits = self._hits + 1 if self.acc >= target_acc else 0
        if self._hits >= patience:
            self.done = True
        return self.done

    def save(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": {
                    "n_in": self.n_in,
                    "n_hidden": self.n_hidden,
                    "n_out": self.n_out,
                    "alpha": self.alpha,
                    "sigma_rec": self.sigma_rec,
                    "activation": self.activation,
                    "rec_gain": self.rec_gain,
                },
                "task": self.task,
                "step": self.step,
                "probe_idx": self.probe_idx.cpu(),
                "val_idx": self.val_idx.cpu(),
            },
            path,
        )
