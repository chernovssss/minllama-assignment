from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):  # type: ignore
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients

                if "first_moment" not in state:
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["second_moment"] = torch.zeros_like(p.data)
                    state["step"] = 0

                state["step"] += 1

                beta1 = group["betas"][0]
                beta2 = group["betas"][1]
                state["first_moment"] = (
                    beta1 * state["first_moment"] + (1 - beta1) * grad
                )
                state["second_moment"] = beta2 * state["second_moment"] + (
                    1 - beta2
                ) * (grad**2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                first_moment = state["first_moment"] / (
                    1 - group["betas"][0] ** state["step"]
                )
                second_moment = state["second_moment"] / (
                    1 - group["betas"][1] ** state["step"]
                )

                # Update parameters
                p.data = p.data - alpha * first_moment / (
                    second_moment.sqrt() + group["eps"]
                )

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if group["weight_decay"] > 0.0:
                    p.data = p.data - alpha * group["weight_decay"] * p.data

        return loss
