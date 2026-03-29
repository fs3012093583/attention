import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        patch = self.patch_size
        patches = x.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, -1, channels * patch * patch)
        return self.proj(patches)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, d_model = tokens.shape

        def reshape_heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(self.q_proj(tokens))
        k = reshape_heads(self.k_proj(tokens))
        v = reshape_heads(self.v_proj(tokens))

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        mixed = torch.matmul(attn, v)
        mixed = mixed.transpose(1, 2).contiguous().view(batch_size, num_tokens, d_model)
        mixed = self.out_proj(mixed)
        return self.proj_drop(mixed)


class StandardMLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class CustomAttentionFFN(nn.Module):
    def __init__(self, d_model: int, slots: int, dropout: float) -> None:
        super().__init__()
        self.slots = slots
        self.expand = nn.Linear(d_model, slots * d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, d_model = tokens.shape
        expanded = torch.nn.functional.gelu(self.expand(tokens))
        expanded = expanded.view(batch_size, num_tokens * self.slots, d_model)

        q = self.q_proj(expanded)
        k = self.k_proj(expanded)
        v = self.v_proj(expanded)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_model), dim=-1)
        mixed = torch.matmul(attn, v)
        mixed = self.o_proj(mixed)
        mixed = self.dropout(mixed)

        return mixed.view(batch_size, num_tokens, self.slots, d_model).mean(dim=2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        ffn_module: nn.Module,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ffn_module

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class ViTClassifier(nn.Module):
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        depth: int,
        num_heads: int,
        num_classes: int,
        dropout: float,
        mlp_ratio: float,
        use_custom_ffn: bool,
        custom_slots: int,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, d_model=d_model)
        num_patches = (32 // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)

        blocks = []
        for _ in range(depth):
            ffn = (
                CustomAttentionFFN(d_model=d_model, slots=custom_slots, dropout=dropout)
                if use_custom_ffn
                else StandardMLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)
            )
            blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    ffn_module=ffn,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        batch_size = tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])


def build_dataloaders(data_root: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    train_set = datasets.CIFAR100(root=str(data_root), train=True, transform=train_transform, download=False)
    test_set = datasets.CIFAR100(root=str(data_root), train=False, transform=test_transform, download=False)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count


@dataclass
class RunResult:
    name: str
    params: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float


def train_model(
    model_name: str,
    model_factory,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> RunResult:
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    train_loss, train_acc = evaluate(model, train_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    return RunResult(
        name=model_name,
        params=count_params(model),
        train_loss=train_loss,
        train_acc=train_acc,
        test_loss=test_loss,
        test_acc=test_acc,
    )


def summarize(results: list[RunResult]) -> str:
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[result.name].append(result)

    lines = []
    lines.append("CIFAR-100 ViT FFN replacement benchmark")
    lines.append("")
    header = (
        f"{'Model':<34} {'Params':>10} {'Train Acc':>12} {'Test Acc':>12} "
        f"{'Train CE':>12} {'Test CE':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, runs in grouped.items():
        params = runs[0].params
        train_acc = torch.tensor([run.train_acc for run in runs], dtype=torch.float64)
        test_acc = torch.tensor([run.test_acc for run in runs], dtype=torch.float64)
        train_loss = torch.tensor([run.train_loss for run in runs], dtype=torch.float64)
        test_loss = torch.tensor([run.test_loss for run in runs], dtype=torch.float64)
        lines.append(
            f"{name:<34} {params:>10} "
            f"{train_acc.mean():>8.4f}±{train_acc.std(unbiased=False):<7.4f} "
            f"{test_acc.mean():>8.4f}±{test_acc.std(unbiased=False):<7.4f} "
            f"{train_loss.mean():>8.4f}±{train_loss.std(unbiased=False):<7.4f} "
            f"{test_loss.mean():>8.4f}±{test_loss.std(unbiased=False):<7.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare standard ViT against a custom attention-style FFN replacement.")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Projects\data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--custom-slots", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    train_loader, test_loader = build_dataloaders(args.data_root, args.batch_size, args.num_workers)

    print(f"Using device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"patch_size={args.patch_size}, d_model={args.d_model}, depth={args.depth}, heads={args.num_heads}")
    print(f"custom_slots={args.custom_slots}")

    configs = [
        (
            "Standard ViT",
            lambda: ViTClassifier(
                patch_size=args.patch_size,
                d_model=args.d_model,
                depth=args.depth,
                num_heads=args.num_heads,
                num_classes=100,
                dropout=args.dropout,
                mlp_ratio=args.mlp_ratio,
                use_custom_ffn=False,
                custom_slots=args.custom_slots,
            ),
        ),
        (
            "ViT with custom attention FFN",
            lambda: ViTClassifier(
                patch_size=args.patch_size,
                d_model=args.d_model,
                depth=args.depth,
                num_heads=args.num_heads,
                num_classes=100,
                dropout=args.dropout,
                mlp_ratio=args.mlp_ratio,
                use_custom_ffn=True,
                custom_slots=args.custom_slots,
            ),
        ),
    ]

    results = []
    for seed in range(args.seeds):
        set_seed(100 + seed)
        for model_name, model_factory in configs:
            result = train_model(
                model_name=model_name,
                model_factory=model_factory,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            results.append(result)
            print(
                f"[seed {100 + seed}] {result.name}: "
                f"params={result.params}, train_acc={result.train_acc:.4f}, test_acc={result.test_acc:.4f}"
            )

    print()
    print(summarize(results))


if __name__ == "__main__":
    main()
