# train.py
import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / len(loader)


def train_model(model, train_loader, device, lr=1e-3, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []

    for e in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        losses.append(loss)
        print(f"Epoch {e+1}/{epochs} | Loss: {loss:.4f}")

    return model, losses
