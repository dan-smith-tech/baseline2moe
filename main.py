import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE = 14
EMBED_DIM = 64
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EXPERTS = 4
SELECT_TOP_K = 2

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
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(EMBED_DIM, NUM_EXPERTS, bias=False)

    def forward(self, x):
        logits = self.gate(x)

        topk_logits, topk_indices = torch.topk(logits, SELECT_TOP_K, dim=-1)
        top_k_probs = torch.softmax(topk_logits, dim=-1)

        weights = torch.zeros_like(logits).to(x.device)
        weights.scatter(1, topk_indices, top_k_probs)

        return weights, topk_indices


class Expert(nn.Module):
    FEEDFORWARD_DIM = 128

    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, self.FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(self.FEEDFORWARD_DIM, EMBED_DIM),
        )

    def forward(self, x):
        return self.ffn(x)


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(NUM_EXPERTS)])
        self.gate = Gate()

    def forward(self, x):
        x = x.view(-1, x.shape[-1])
        gate_weights, topk_indices = self.gate(x)

        final_output = torch.zeros(x.shape(0), EMBED_DIM, dtype=x.dtype).to(x.device)

        expert_outputs = []


class Baseline(nn.Module):
    FEEDFORWARD_DIM = 128

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

        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, self.FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(self.FEEDFORWARD_DIM, EMBED_DIM),
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


loss_fn = nn.CrossEntropyLoss()

baseline_model = Baseline().to(DEVICE)
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    baseline_optimizer, T_max=EPOCHS
)

print(f"Using device: {DEVICE}\n")

for epoch in range(EPOCHS):
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
        f"Epoch {epoch + 1}/{EPOCHS}:\n"
        f"Baseline -- Train Loss: {baseline_train_loss:.4f}, Test Loss: {baseline_test_loss:.4f}, Test Accuracy: {baseline_test_accuracy:.4f}\n"
    )
