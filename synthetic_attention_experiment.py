import math
from dataclasses import dataclass

import autograd.numpy as np
from autograd import grad


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def glorot(rng, fan_in, fan_out):
    scale = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-scale, scale, size=(fan_in, fan_out))


def zeros(*shape):
    return np.zeros(shape)


def count_params(params):
    return int(sum(np.prod(p.shape) for p in params))


def flatten_params(params):
    return [p for p in params]


def attention_block(tokens, w_q, w_k, w_v, w_o):
    q = np.einsum("bnd,df->bnf", tokens, w_q)
    k = np.einsum("bnd,df->bnf", tokens, w_k)
    v = np.einsum("bnd,df->bnf", tokens, w_v)
    scores = np.einsum("bnd,bmd->bnm", q, k) / math.sqrt(tokens.shape[-1])
    weights = softmax(scores, axis=-1)
    mixed = np.einsum("bnm,bmd->bnd", weights, v)
    return np.einsum("bnd,df->bnf", mixed, w_o)


def init_ffn(rng, d, n):
    hidden = n * d
    return [
        glorot(rng, d, hidden),
        zeros(hidden),
        glorot(rng, hidden, 1),
        zeros(1),
    ]


def ffn_forward(params, x):
    w1, b1, w2, b2 = params
    hidden = gelu(np.dot(x, w1) + b1)
    return np.dot(hidden, w2) + b2


def init_attn(rng, d, n, use_activation):
    params = [
        glorot(rng, d, n * d),
        zeros(n * d),
        glorot(rng, d, d),
        glorot(rng, d, d),
        glorot(rng, d, d),
        glorot(rng, d, d),
        glorot(rng, d, 1),
        zeros(1),
    ]
    if use_activation:
        return params
    return params


def attn_forward(params, x, n, use_activation):
    w_expand, b_expand, w_q, w_k, w_v, w_o, w_head, b_head = params
    expanded = np.dot(x, w_expand) + b_expand
    if use_activation:
        expanded = gelu(expanded)
    tokens = expanded.reshape((x.shape[0], n, x.shape[1]))
    mixed = attention_block(tokens, w_q, w_k, w_v, w_o)
    pooled = np.mean(mixed, axis=1)
    return np.dot(pooled, w_head) + b_head


def build_task_params(rng, d, n):
    return {
        "slot_proj": glorot(rng, d, n * d),
        "pair_matrix": rng.normal(scale=0.35, size=(n, n)),
        "out_proj": rng.normal(scale=0.4, size=(d,)),
    }


def sample_interaction_task(rng, num_samples, d, n, task_params):
    x = rng.normal(size=(num_samples, d))
    slots = gelu(np.dot(x, task_params["slot_proj"])).reshape((num_samples, n, d))
    slot_strength = np.einsum("bnd,d->bn", slots, task_params["out_proj"])
    pair_scores = np.einsum("bi,ij,bj->b", slot_strength, task_params["pair_matrix"], slot_strength)
    pooled = np.mean(slots, axis=1)
    y = pair_scores + 0.3 * np.sum(pooled[:, : d // 2] ** 2, axis=1)
    y = y[:, None]
    return x, y


@dataclass
class AdamState:
    m: list
    v: list
    t: int = 0


def adam_init(params):
    return AdamState(
        m=[np.zeros_like(p) for p in params],
        v=[np.zeros_like(p) for p in params],
        t=0,
    )


def adam_step(params, grads, state, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    state.t += 1
    new_params = []
    for i, (p, g) in enumerate(zip(params, grads)):
        state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g
        state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * (g * g)
        m_hat = state.m[i] / (1.0 - beta1 ** state.t)
        v_hat = state.v[i] / (1.0 - beta2 ** state.t)
        new_params.append(p - lr * m_hat / (np.sqrt(v_hat) + eps))
    return new_params, state


def mse(pred, target):
    diff = pred - target
    return np.mean(diff * diff)


def train_model(name, init_fn, forward_fn, train_x, train_y, val_x, val_y, steps, seed):
    rng = np.random.RandomState(seed)
    params = flatten_params(init_fn(rng))

    def loss_fn(current_params, batch_x, batch_y):
        pred = forward_fn(current_params, batch_x)
        return mse(pred, batch_y)

    loss_grad = grad(loss_fn)
    state = adam_init(params)

    for _ in range(steps):
        grads = loss_grad(params, train_x, train_y)
        params, state = adam_step(params, grads, state)

    train_loss = float(loss_fn(params, train_x, train_y))
    val_loss = float(loss_fn(params, val_x, val_y))
    return {
        "name": name,
        "params": count_params(params),
        "train_mse": train_loss,
        "val_mse": val_loss,
    }


def summarize(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["name"], []).append(result)

    lines = []
    lines.append("Synthetic task: latent slot interaction regression")
    lines.append("")
    header = f"{'Model':<28} {'Params':>8} {'Train MSE':>12} {'Val MSE':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, runs in grouped.items():
        params = runs[0]["params"]
        train_mean = np.mean([r["train_mse"] for r in runs])
        val_mean = np.mean([r["val_mse"] for r in runs])
        train_std = np.std([r["train_mse"] for r in runs])
        val_std = np.std([r["val_mse"] for r in runs])
        lines.append(
            f"{name:<28} {params:>8} {train_mean:>8.4f}±{train_std:<7.4f} {val_mean:>8.4f}±{val_std:<7.4f}"
        )
    return "\n".join(lines)


def main():
    d = 16
    n = 4
    task_params = build_task_params(np.random.RandomState(3), d, n)
    train_x, train_y = sample_interaction_task(np.random.RandomState(7), 512, d, n, task_params)
    val_x, val_y = sample_interaction_task(np.random.RandomState(11), 512, d, n, task_params)

    train_mean = np.mean(train_y)
    train_std = np.std(train_y) + 1e-6
    train_y = (train_y - train_mean) / train_std
    val_y = (val_y - train_mean) / train_std

    configs = [
        (
            "FFN d->nd->1",
            lambda rng: init_ffn(rng, d, n),
            lambda params, x: ffn_forward(params, x),
        ),
        (
            "Expand->Attn->Pool->1",
            lambda rng: init_attn(rng, d, n, use_activation=False),
            lambda params, x: attn_forward(params, x, n=n, use_activation=False),
        ),
        (
            "Expand->GELU->Attn->Pool->1",
            lambda rng: init_attn(rng, d, n, use_activation=True),
            lambda params, x: attn_forward(params, x, n=n, use_activation=True),
        ),
    ]

    results = []
    for seed in range(5):
        for name, init_fn, forward_fn in configs:
            results.append(
                train_model(
                    name=name,
                    init_fn=init_fn,
                    forward_fn=forward_fn,
                    train_x=train_x,
                    train_y=train_y,
                    val_x=val_x,
                    val_y=val_y,
                    steps=400,
                    seed=100 + seed,
                )
            )

    print(summarize(results))


if __name__ == "__main__":
    main()
