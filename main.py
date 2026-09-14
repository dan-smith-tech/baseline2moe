import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE = 14
EMBED_DIM = 64
FEEDFORWARD_DIM = 128
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.0

# these mirror the moe_* config block given for the big model, only scaled
# down where needed so the standalone test trains quickly
MOE_BASE_NUM_EXPERTS = 8
MOE_EXPERT_SEGMENTATION_FACTOR = 1
MOE_BASE_SELECT_TOP_K = 2
MOE_NUM_SHARED_EXPERTS = 0
MOE_SCALE_EXPERT_DIM = True
MOE_ALPHA = 0.01
MOE_CZ = 0.001
MOE_USE_ROUTER_NOISE = True

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class Gate(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        select_top_k: int,
        use_router_noise: bool,
    ) -> None:
        super().__init__()
        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        self.noise_router = (
            nn.Linear(embed_dim, num_experts, bias=False) if use_router_noise else None
        )
        self.select_top_k = select_top_k

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        router_logits = self.router(x)
        noisy_router_logits = router_logits

        # noisy top-k routing during training to keep exploration healthy
        if self.training and self.noise_router is not None:
            noise_std = F.softplus(self.noise_router(x))
            noisy_router_logits = (
                noisy_router_logits + torch.randn_like(noisy_router_logits) * noise_std
            )

        # get top-k expert scores per object
        topk_logits, topk_indices = torch.topk(
            noisy_router_logits, self.select_top_k, dim=-1
        )
        # probability distribution over selected experts only
        topk_weights = torch.softmax(topk_logits, dim=-1)

        # dense gate weights over all experts from clean logits; used for losses/stats
        dense_gate_weights = torch.softmax(router_logits, dim=-1)

        return router_logits, dense_gate_weights, topk_weights, topk_indices


class Expert(nn.Module):
    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            # GELU activation maintained from original PET FFN
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class MoE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_dim: int,
        base_num_experts: int,
        base_select_top_k: int,
        num_shared_experts: int,
        expert_segmentation_factor: int,
        scale_expert_dim: bool,
        alpha: float,
        c_z: float,
        use_router_noise: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.base_num_experts = base_num_experts
        self.base_select_top_k = base_select_top_k
        self.expert_segmentation_factor = expert_segmentation_factor
        self.num_shared_experts = num_shared_experts
        self.alpha = alpha
        self.c_z = c_z

        total_experts = self.base_num_experts * self.expert_segmentation_factor
        # num_experts is the total budget - routed experts fill the remainder after reserving shared slots
        self.num_experts = total_experts - num_shared_experts
        self.select_top_k = self.base_select_top_k * self.expert_segmentation_factor
        # when scale_expert_dim is True divide each expert's hidden dim by select_top_k
        # to keep per-token compute constant vs. a vanilla FFN; even shared experts
        # are impacted by segmentation scaling of k
        self.expert_hidden_dim = (
            int(feedforward_dim / (self.select_top_k + self.num_shared_experts))
            if scale_expert_dim
            else feedforward_dim
        )

        self.gate = Gate(
            embed_dim,
            self.num_experts,
            self.select_top_k,
            use_router_noise=use_router_noise,
        )
        self.routed_experts = nn.ModuleList(
            [
                Expert(embed_dim, self.expert_hidden_dim, dropout)
                for _ in range(self.num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(embed_dim, self.expert_hidden_dim, dropout)
                for _ in range(self.num_shared_experts)
            ]
        )

        # running tally of how many objects each expert has processed, kept only
        # for the expert distribution feature and reset by the training loop
        self.register_buffer("expert_counts", torch.zeros(self.num_experts))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = x.shape
        # collapse batch/object axes to a 2D tensor so that each row corresponds to a single object to route to experts
        x = x.reshape(-1, x.shape[-1])
        num_objects = x.shape[0]

        router_logits, dense_gate_weights, topk_weights, topk_indices = self.gate(x)

        routed_output = torch.zeros(
            (num_objects, self.embed_dim), dtype=x.dtype, device=x.device
        )
        objects_per_expert = torch.zeros(
            self.num_experts, dtype=torch.long, device=x.device
        )

        if num_objects > 0:
            # get flat list of each object id repeated for each of its top-k experts - e.g., [0, 0, 1, 1, 2, 2, ...]
            object_indices = (
                torch.arange(num_objects, device=x.device)
                .unsqueeze(1)
                .expand(-1, self.select_top_k)
                .reshape(-1)
            )
            # get flat list of which expert each object is assigned to - of size [num_objects * top_k]
            expert_indices = topk_indices.reshape(-1)
            # get flat list of corresponding expert weights for each object - of size [num_objects * top_k]
            expert_weights = topk_weights.reshape(-1)

            # sort by expert index so that all objects for each expert are grouped together
            order = torch.argsort(expert_indices)
            object_indices = object_indices[order]
            expert_indices = expert_indices[order]
            expert_weights = expert_weights[order]
            # count how many objects are assigned to each expert to know how to split the input tensor for each expert's forward pass
            objects_per_expert = torch.bincount(
                expert_indices, minlength=self.num_experts
            )

            with torch.no_grad():
                self.expert_counts += objects_per_expert.to(self.expert_counts.device)

            cursor = 0
            # iterate through each expert's assigned objects in order of expert index
            for expert_id, count in enumerate(objects_per_expert.tolist()):
                if count == 0:
                    continue
                end = cursor + count

                current_object_indices = object_indices[cursor:end]
                current_inputs = x.index_select(0, current_object_indices)

                current_outputs = self.routed_experts[expert_id](current_inputs)
                current_weights = expert_weights[cursor:end].unsqueeze(-1)

                routed_output.index_add_(
                    0, current_object_indices, current_outputs * current_weights
                )

                cursor = end

        # shared experts are not part of the routing decisions - every object passes through all of them
        if self.num_shared_experts > 0:
            shared_output = torch.zeros_like(routed_output)
            for shared_expert in self.shared_experts:
                shared_output = shared_output + shared_expert(x)
            final_output = routed_output + shared_output
        else:
            final_output = routed_output

        # fi - proportion of objects assigned to each expert
        denom = max(num_objects * self.select_top_k, 1)
        dispatch_fraction = (
            objects_per_expert.to(dtype=dense_gate_weights.dtype) / denom
        )
        # pi - average probability of each expert being selected across all objects
        mean_router_prob = (
            dense_gate_weights.mean(dim=0)
            if num_objects > 0
            else torch.zeros_like(dispatch_fraction)
        )
        # l_aux is the sum across experts of fi * pi, scaled by alpha and num_experts
        l_aux = (
            self.alpha
            * self.num_experts
            * torch.sum(dispatch_fraction * mean_router_prob)
        )

        if num_objects > 0:
            # use clean router logits without noise (pre-softmax)
            cz_lz = self.c_z * torch.mean(torch.logsumexp(router_logits, dim=-1).pow(2))
        else:
            cz_lz = torch.zeros((), dtype=x.dtype, device=x.device)

        final_output = final_output.view(
            original_shape[0], original_shape[1], self.embed_dim
        )

        return final_output, l_aux, cz_lz

    def get_expert_distribution(self):
        total = self.expert_counts.sum()
        if total == 0:
            return torch.zeros_like(self.expert_counts)
        return (self.expert_counts / total).cpu()

    def reset_expert_counts(self):
        self.expert_counts.zero_()


class Model(nn.Module):
    def __init__(self, moe=False):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            1, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, (28 // PATCH_SIZE) ** 2 + 1, EMBED_DIM)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM, num_heads=1, batch_first=True
        )

        self.norm1 = nn.LayerNorm(EMBED_DIM)

        if moe:
            self.ffn = MoE(
                embed_dim=EMBED_DIM,
                feedforward_dim=FEEDFORWARD_DIM,
                base_num_experts=MOE_BASE_NUM_EXPERTS,
                base_select_top_k=MOE_BASE_SELECT_TOP_K,
                num_shared_experts=MOE_NUM_SHARED_EXPERTS,
                expert_segmentation_factor=MOE_EXPERT_SEGMENTATION_FACTOR,
                scale_expert_dim=MOE_SCALE_EXPERT_DIM,
                alpha=MOE_ALPHA,
                c_z=MOE_CZ,
                use_router_noise=MOE_USE_ROUTER_NOISE,
                dropout=DROPOUT,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(EMBED_DIM, FEEDFORWARD_DIM),
                nn.ReLU(),
                nn.Linear(FEEDFORWARD_DIM, EMBED_DIM),
            )

        self.norm2 = nn.LayerNorm(EMBED_DIM)

        self.classifier = nn.Linear(EMBED_DIM, 10)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        if isinstance(self.ffn, MoE):
            ffn_output, l_aux, cz_lz = self.ffn(x)
            x = self.norm2(x + ffn_output)
            logits = self.classifier(x[:, 0])
            return logits, l_aux, cz_lz
        else:
            ffn_output = self.ffn(x)
            x = self.norm2(x + ffn_output)
            return self.classifier(x[:, 0])


def train(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(loader)
    return avg_loss


def train_moe(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output, l_aux, cz_lz = model(data)
        loss = loss_fn(output, target) + l_aux + cz_lz
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(loader)
    return avg_loss


def test(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def test_moe(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output, l_aux, cz_lz = model(data)
            loss = loss_fn(output, target) + l_aux + cz_lz
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


loss_fn = nn.CrossEntropyLoss()

baseline_model = Model().to(DEVICE)
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    baseline_optimizer, T_max=EPOCHS
)

moe_model = Model(moe=True).to(DEVICE)
moe_optimizer = torch.optim.Adam(moe_model.parameters(), lr=0.001)
moe_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(moe_optimizer, T_max=EPOCHS)

print(f"Using device: {DEVICE}\n")

num_experts = moe_model.ffn.num_experts
expert_distribution_history = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    baseline_train_loss = train(
        baseline_model,
        train_loader,
        baseline_optimizer,
        baseline_scheduler,
        loss_fn,
    )
    baseline_test_loss, baseline_test_accuracy = test(
        baseline_model, test_loader, loss_fn
    )
    print(
        f"Baseline -- Train Loss: {baseline_train_loss:.4f}, Test Loss: {baseline_test_loss:.4f}, Test Accuracy: {baseline_test_accuracy:.4f}"
    )

    moe_train_loss = train_moe(
        moe_model, train_loader, moe_optimizer, moe_scheduler, loss_fn
    )
    moe_test_loss, moe_test_accuracy = test_moe(moe_model, test_loader, loss_fn)
    print(
        f"MoE      -- Train Loss: {moe_train_loss:.4f}, Test Loss: {moe_test_loss:.4f}, Test Accuracy: {moe_test_accuracy:.4f}"
    )

    distribution = moe_model.ffn.get_expert_distribution()
    expert_distribution_history.append(distribution.numpy())
    distribution_str = " ".join(
        f"E{i}: {p * 100:.1f}%" for i, p in enumerate(distribution)
    )
    print(f"Expert distribution -- {distribution_str}\n")
    moe_model.ffn.reset_expert_counts()

fig, ax = plt.subplots(figsize=(8, 3))
history = list(zip(*expert_distribution_history))
for expert_index, shares in enumerate(history):
    ax.plot(range(1, EPOCHS + 1), shares, label=f"Expert {expert_index}")

ax.set_xlabel("Epoch")
ax.set_ylabel("Share of tokens routed")
ax.set_title("Complex MoE — Expert load distribution over training")
ax.legend(ncol=4, fontsize=7)
fig.tight_layout()
fig.savefig("expert_distribution_complex.png", dpi=150)
print("Saved expert_distribution_complex.png")
