import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def ffn_param_count(d: int, hidden: int) -> int:
    return (d + 1) * hidden + (hidden + 1)


def find_matched_ffn_hidden(d: int, target_params: int) -> int:
    best_hidden = 1
    best_gap = abs(ffn_param_count(d, best_hidden) - target_params)
    hidden = 2
    while True:
        current_params = ffn_param_count(d, hidden)
        current_gap = abs(current_params - target_params)
        if current_gap < best_gap:
            best_hidden = hidden
            best_gap = current_gap
        if current_params > target_params and current_gap > best_gap:
            break
        hidden += 1
    return best_hidden


class FFNRegressor(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpandAttnPoolRegressor(nn.Module):
    def __init__(
        self,
        d: int,
        n: int,
        use_activation: bool,
        use_residual: bool = False,
        learn_value: bool = True,
        use_output_proj: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.n = n
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.learn_value = learn_value
        self.use_output_proj = use_output_proj

        self.expand = nn.Linear(d, n * d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d) if learn_value else None
        self.o_proj = nn.Linear(d, d) if use_output_proj else None
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(x)
        if self.use_activation:
            expanded = F.gelu(expanded)
        tokens = expanded.view(x.size(0), self.n, self.d)

        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens) if self.learn_value else tokens
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d), dim=-1)
        mixed = torch.matmul(attn, v)
        if self.use_output_proj:
            mixed = self.o_proj(mixed)
        pooled = mixed.mean(dim=1)

        if self.use_residual:
            pooled = pooled + x
        return self.head(pooled)


@dataclass
class TaskParams:
    slot_proj: torch.Tensor
    pair_matrix: torch.Tensor
    out_proj: torch.Tensor


def build_task_params(d: int, n: int, seed: int, device: torch.device) -> TaskParams:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    slot_proj = torch.empty(d, n * d, device=device)
    nn.init.xavier_uniform_(slot_proj, gain=1.0)
    pair_matrix = torch.randn(n, n, generator=generator, device=device) * 0.35
    out_proj = torch.randn(d, generator=generator, device=device) * 0.4
    return TaskParams(slot_proj=slot_proj, pair_matrix=pair_matrix, out_proj=out_proj)


def sample_interaction_task(
    num_samples: int,
    d: int,
    n: int,
    task_params: TaskParams,
    seed: int,
    device: torch.device,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(num_samples, d, generator=generator, device=device)
    slots = F.gelu(x @ task_params.slot_proj).view(num_samples, n, d)
    slot_strength = torch.einsum("bnd,d->bn", slots, task_params.out_proj)
    pair_scores = torch.einsum("bi,ij,bj->b", slot_strength, task_params.pair_matrix, slot_strength)
    pooled = slots.mean(dim=1)
    y = pair_scores + 0.3 * pooled[:, : d // 2].square().sum(dim=1)
    return x, y.unsqueeze(-1)


def normalize_targets(train_y: torch.Tensor, val_y: torch.Tensor):
    mean = train_y.mean()
    std = train_y.std().clamp_min(1e-6)
    return (train_y - mean) / std, (val_y - mean) / std


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            pred = model(batch_x.to(device))
            loss = F.mse_loss(pred, batch_y.to(device), reduction="sum")
            total_loss += loss.item()
            total_count += batch_x.size(0)
    return total_loss / total_count


def train_model(
    model_name: str,
    model_factory,
    train_loader: DataLoader,
    val_loader: DataLoader,
    steps: int,
    lr: float,
    device: torch.device,
):
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_iter = iter(train_loader)

    model.train()
    for _ in range(steps):
        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_y = next(train_iter)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(batch_x)
        loss = F.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()

    return {
        "name": model_name,
        "params": count_params(model),
        "train_mse": evaluate(model, train_loader, device),
        "val_mse": evaluate(model, val_loader, device),
    }


def summarize(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["name"], []).append(result)

    lines = []
    lines.append("Synthetic task: latent slot interaction regression")
    lines.append("")
    header = f"{'Model':<34} {'Params':>8} {'Train MSE':>12} {'Val MSE':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, runs in grouped.items():
        params = runs[0]["params"]
        train_values = torch.tensor([r["train_mse"] for r in runs], dtype=torch.float64)
        val_values = torch.tensor([r["val_mse"] for r in runs], dtype=torch.float64)
        lines.append(
            f"{name:<34} {params:>8} "
            f"{train_values.mean():>8.4f}±{train_values.std(unbiased=False):<7.4f} "
            f"{val_values.mean():>8.4f}±{val_values.std(unbiased=False):<7.4f}"
        )
    return "\n".join(lines)


def main():
    device = get_device()
    print(f"Using device: {device}")
    d = 16
    n = 4
    train_size = 2048
    val_size = 2048
    batch_size = 128
    steps = 800
    lr = 3e-3
    num_seeds = 5

    set_seed(3)
    task_params = build_task_params(d=d, n=n, seed=3, device=device)
    train_x, train_y = sample_interaction_task(train_size, d, n, task_params, seed=7, device=device)
    val_x, val_y = sample_interaction_task(val_size, d, n, task_params, seed=11, device=device)
    train_y, val_y = normalize_targets(train_y, val_y)

    train_loader = DataLoader(
        TensorDataset(train_x.cpu(), train_y.cpu()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x.cpu(), val_y.cpu()),
        batch_size=batch_size,
        shuffle=False,
    )

    target_param_model = ExpandAttnPoolRegressor(d=d, n=n, use_activation=True, use_residual=True)
    matched_ffn_hidden = find_matched_ffn_hidden(d=d, target_params=count_params(target_param_model))

    configs = [
        (f"FFN d->{matched_ffn_hidden}->1", lambda: FFNRegressor(d=d, hidden=matched_ffn_hidden)),
        ("Expand->Attn->Pool->1", lambda: ExpandAttnPoolRegressor(d=d, n=n, use_activation=False)),
        ("Expand->GELU->Attn->Pool->1", lambda: ExpandAttnPoolRegressor(d=d, n=n, use_activation=True)),
        (
            "Expand->GELU->Attn->Pool->1 + residual",
            lambda: ExpandAttnPoolRegressor(d=d, n=n, use_activation=True, use_residual=True),
        ),
        (
            "Expand->GELU->Attn(V=T)->Pool->1",
            lambda: ExpandAttnPoolRegressor(d=d, n=n, use_activation=True, learn_value=False),
        ),
        (
            "Expand->GELU->Attn(V=T,no Wo)->Pool->1",
            lambda: ExpandAttnPoolRegressor(
                d=d,
                n=n,
                use_activation=True,
                learn_value=False,
                use_output_proj=False,
            ),
        ),
    ]

    results = []
    for seed in range(num_seeds):
        set_seed(100 + seed)
        for model_name, model_factory in configs:
            results.append(
                train_model(
                    model_name=model_name,
                    model_factory=model_factory,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    steps=steps,
                    lr=lr,
                    device=device,
                )
            )

    print(f"Device: {device}")
    print(f"Matched FFN hidden width: {matched_ffn_hidden}")
    print(summarize(results))


if __name__ == "__main__":
    main()
