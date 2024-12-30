from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, device="cpu"):
    model.train()
    loop = tqdm(dataloader, desc=f"Epoch {epoch}/{dataloader.dataset.__len__()}")

    total_loss = 0
    for batch_idx, (images, targets) in enumerate(loop):
        images = images.to(device)
        targets = targets.to(device)

        # Forward
        predictions = model(images)
        loss = criterion(predictions, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}] -> Loss: {avg_loss:.4f}")
