import torch
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp=True):
    model.train()
    total_loss = 0.0

    scaler = GradScaler(enabled=use_amp)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model,
    train_loader,
    device,
    lr=1e-4,
    epochs=10,
    class_weights=None,
    use_amp=True,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    history = {"loss": []}

    for epoch in range(epochs):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp
        )
        history["loss"].append(loss)
        print(f"[Epoch {epoch+1}/{epochs}] loss = {loss:.4f}")

    return model, history
