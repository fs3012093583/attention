import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"Image size {(height, width)} must be divisible by patch size {patch}.")

        patches = x.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, -1, channels * patch * patch)
        return self.proj(patches)


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, use_activation: bool, use_residual: bool) -> None:
        super().__init__()
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        if self.use_activation:
            tokens = F.gelu(tokens)

        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(tokens.size(-1)), dim=-1)
        mixed = torch.matmul(attn, v)
        mixed = self.o_proj(mixed)

        if self.use_residual:
            mixed = mixed + residual
        return mixed


class FlattenMLPClassifier(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        d_model: int,
        hidden: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, d_model=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(num_patches * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        return self.classifier(tokens.flatten(start_dim=1))


class PatchAttentionClassifier(nn.Module):
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        num_classes: int,
        num_layers: int,
        use_activation: bool,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, d_model=d_model)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    d_model=d_model,
                    use_activation=use_activation,
                    use_residual=use_residual,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


def mlp_classifier_param_count(input_dim: int, hidden: int, num_classes: int) -> int:
    return (input_dim + 1) * hidden + (hidden + 1) * num_classes


def find_matched_hidden(input_dim: int, target_params: int, num_classes: int) -> int:
    best_hidden = 1
    best_gap = abs(mlp_classifier_param_count(input_dim, best_hidden, num_classes) - target_params)
    hidden = 2
    while True:
        current_params = mlp_classifier_param_count(input_dim, hidden, num_classes)
        current_gap = abs(current_params - target_params)
        if current_gap < best_gap:
            best_hidden = hidden
            best_gap = current_gap
        if current_params > target_params and current_gap > best_gap:
            break
        hidden += 1
    return best_hidden


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
            loss = F.cross_entropy(logits, labels, reduction="sum")
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
            loss = F.cross_entropy(logits, labels)
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
    lines.append("CIFAR-100 classification benchmark")
    lines.append("")
    header = (
        f"{'Model':<34} {'Params':>8} {'Train Acc':>12} {'Test Acc':>12} "
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
            f"{name:<34} {params:>8} "
            f"{train_acc.mean():>8.4f}±{train_acc.std(unbiased=False):<7.4f} "
            f"{test_acc.mean():>8.4f}±{test_acc.std(unbiased=False):<7.4f} "
            f"{train_loss.mean():>8.4f}±{train_loss.std(unbiased=False):<7.4f} "
            f"{test_loss.mean():>8.4f}±{test_loss.std(unbiased=False):<7.4f}"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MLP and attention-style classifiers on CIFAR-100.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Projects\data"),
        help="Root directory passed to torchvision.datasets.CIFAR100.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--deep-layers", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if 32 % args.patch_size != 0:
        raise ValueError("--patch-size must divide 32 for CIFAR-100 images.")

    device = get_device()
    num_classes = 100
    patches_per_side = 32 // args.patch_size
    num_patches = patches_per_side * patches_per_side
    input_dim = num_patches * args.d_model

    train_loader, test_loader = build_dataloaders(args.data_root, args.batch_size, args.num_workers)

    target_model = PatchAttentionClassifier(
        patch_size=args.patch_size,
        d_model=args.d_model,
        num_classes=num_classes,
        num_layers=1,
        use_activation=True,
        use_residual=True,
    )
    matched_hidden = find_matched_hidden(input_dim, count_params(target_model), num_classes)

    print(f"Using device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Patch size: {args.patch_size}")
    print(f"Patch tokens: {num_patches}")
    print(f"Deep attention layers: {args.deep_layers}")
    print(f"Matched MLP hidden width: {matched_hidden}")

    configs = [
        (
            f"MLP flatten d->{matched_hidden}->100",
            lambda: FlattenMLPClassifier(
                patch_size=args.patch_size,
                num_patches=num_patches,
                d_model=args.d_model,
                hidden=matched_hidden,
                num_classes=num_classes,
            ),
        ),
        (
            "Patch->Attn->Pool->100",
            lambda: PatchAttentionClassifier(
                patch_size=args.patch_size,
                d_model=args.d_model,
                num_classes=num_classes,
                num_layers=1,
                use_activation=False,
            ),
        ),
        (
            "Patch->GELU->Attn->Pool->100",
            lambda: PatchAttentionClassifier(
                patch_size=args.patch_size,
                d_model=args.d_model,
                num_classes=num_classes,
                num_layers=1,
                use_activation=True,
            ),
        ),
        (
            "Patch->GELU->Attn->Pool->100 + residual",
            lambda: PatchAttentionClassifier(
                patch_size=args.patch_size,
                d_model=args.d_model,
                num_classes=num_classes,
                num_layers=1,
                use_activation=True,
                use_residual=True,
            ),
        ),
        (
            f"Patch->GELU->Attn x{args.deep_layers}->Pool->100 + residual",
            lambda: PatchAttentionClassifier(
                patch_size=args.patch_size,
                d_model=args.d_model,
                num_classes=num_classes,
                num_layers=args.deep_layers,
                use_activation=True,
                use_residual=True,
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
                f"train_acc={result.train_acc:.4f}, test_acc={result.test_acc:.4f}"
            )

    print()
    print(summarize(results))


if __name__ == "__main__":
    main()
