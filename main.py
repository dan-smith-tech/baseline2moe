import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 5


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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 7
        self.embed_dim = 64

        self.patch_embedding = nn.Conv2d(
            1, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, (28 // self.patch_size) ** 2 + 1, self.embed_dim)
        )

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim, nhead=8, dim_feedforward=256
                )
                for _ in range(6)
            ]
        )

        self.classifier = nn.Linear(self.embed_dim, 10)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        for layer in self.encoder_layers:
            x = layer(x)

        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        return logits


def train(model, loader, optimizer, loss_fn):
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


basic_model = BasicModel().to(DEVICE)
basic_optimizer = torch.optim.Adam(basic_model.parameters(), lr=0.001)

transformer_model = TransformerModel().to(DEVICE)
transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    basic_train_loss = train(basic_model, train_loader, basic_optimizer, loss_fn)
    basic_test_loss, basic_test_accuracy = test(basic_model, test_loader, loss_fn)
    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {basic_train_loss:.4f}, Test Loss: {basic_test_loss:.4f}, Test Accuracy: {basic_test_accuracy:.4f}"
    )

    transformer_train_loss = train(
        transformer_model, train_loader, transformer_optimizer, loss_fn
    )
    transformer_test_loss, transformer_test_accuracy = test(
        transformer_model, test_loader, loss_fn
    )
    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {transformer_train_loss:.4f}, Test Loss: {transformer_test_loss:.4f}, Test Accuracy: {transformer_test_accuracy:.4f}"
    )
