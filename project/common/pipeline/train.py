import torch
import time

# AMP only if CUDA is available
AMP_AVAILABLE = torch.cuda.is_available()
if AMP_AVAILABLE:
    from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss = 0.0

    use_amp = AMP_AVAILABLE  # disable AMP on CPU
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    epoch_start = time.time()
    num_batches = len(loader)

    print(f"\n🚀 Starting epoch {epoch}/{epochs} ({num_batches} batches)")

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # ---- progress print every 20 steps ----
        if (i + 1) % 20 == 0 or (i + 1) == num_batches:
            elapsed = time.time() - epoch_start
            avg_step = elapsed / (i + 1)
            remaining = avg_step * (num_batches - (i + 1))
            print(
                f"  Batch {i+1}/{num_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"ETA: {remaining:.1f}s"
            )

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches

    print(
        f"✅ Epoch {epoch}/{epochs} finished | Loss={avg_loss:.4f} | Time={epoch_time:.1f}s"
    )

    return avg_loss


def train_model(
    model,
    train_loader,
    device,
    lr=1e-4,
    epochs=10,
    class_weights=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss()

    history = {"loss": []}

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epochs
        )
        history["loss"].append(loss)

    return model, history
