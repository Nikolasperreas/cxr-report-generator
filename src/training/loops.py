"""Training and evaluation loops, shared by every PyTorch model (SmallCNN now, ResNet/MobileNet later)."""

import copy
import time

import torch
from torch import nn

from src.evaluation.metrics_cls import classification_metrics


# one epoch is to do a forward backward pass from the whole dataset
# pass from all the batches. lets say we have 160 and update the weights(gradient descent)
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the loader, updating the weights after every batch.
    Returns the epoch's average loss per example (float)."""
    # 1. training mode (BatchNorm uses each batch's statistics)
    model.train()
    # 2. a running total of the loss, starting at 0.0
    total_loss = 0.0  # starts at zero once per epoch

    # 3. for each batch of the loader:
    for images, labels in loader:
        #        move images and labels to the device
        images, labels = images.to(device), labels.to(device)
        #        the 5 lines: zero_grad, forward, loss, backward, step
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        #  add this batch's loss to the total, weighted by its size   (see the toy)
        total_loss += loss.item() * images.shape[0]

    # 4. return the total divided by the number of examples
    return total_loss / len(loader.dataset)


def recalibrate_batchnorm(model, loader, device):
    """Recompute every BatchNorm layer's stored mean and variance for the model's current
    weights: one pass over loader, no weight updates. Call it right before evaluate.
    loader must give un-augmented train images (never val: the model would learn from val)."""
    # 1. find the BatchNorm layers (model.modules() also looks inside nested blocks)
    bn_layers = [
        layer for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)
    ]
    old_momentum = [layer.momentum for layer in bn_layers]
    # 2. forget the stored statistics; momentum None = plain average, not a moving average
    for layer in bn_layers:
        layer.reset_running_stats()
        layer.momentum = None
    # 3. show every image once: train mode so BatchNorm records the statistics,
    #    no backward and no optimizer step so the weights stay exactly the same
    model.train()
    with torch.no_grad():
        for images, _ in loader:
            model(images.to(device))
    # 4. put the momentum back, so the next epoch trains exactly as before
    for layer, momentum in zip(bn_layers, old_momentum):
        layer.momentum = momentum


def evaluate(model, loader, loss_fn, device):
    """Run the model over the loader without changing it.
    Returns (average loss per example, y_true (m, 14), y_prob (m, 14)); the arrays are numpy."""
    # 1. evaluation mode (BatchNorm uses the running statistics from training)
    model.eval()
    # 2. a running total of the loss, and two empty lists: labels, probabilities
    total_loss = 0.0
    all_labels = []
    all_probs = []
    # 3. with torch.no_grad():
    #        for each batch of the loader:
    #            move images and labels to the device
    #            forward -> logits; loss -> add to the total, weighted by batch size
    #            probabilities = sigmoid of the logits
    #            append labels and probabilities to their lists, moved to the CPU
    with torch.no_grad():
        for images, labels in loader:
            #        move images and labels to the device
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.shape[0]
            probs = torch.sigmoid(logits)
            all_labels.append(
                labels.cpu()
            )  # this batch's labels, to the CPU, onto the list
            all_probs.append(probs.cpu())
    # 4. glue each list into one tensor (torch.cat), then turn it into numpy
    y_true = torch.cat(all_labels).numpy()  # 4 spaces: after both blocks
    y_prob = torch.cat(all_probs).numpy()
    return total_loss / len(loader.dataset), y_true, y_prob
    # 5. return the total / number of examples, y_true, y_prob


def fit(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    epochs,
    threshold,
    categories,
    recalibrate_loader=None,
):
    """Train for `epochs`, evaluating val after every epoch; keep the weights of the epoch with
    the lowest val loss. recalibrate_loader: un-augmented train images -> BatchNorm statistics
    recomputed before every val evaluation; None -> skipped (a model whose BatchNorm is frozen).
    Returns (best_state, best_epoch, best_val_loss, history); history = one dict per epoch."""
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        if recalibrate_loader is not None:
            recalibrate_batchnorm(
                model, recalibrate_loader, device
            )  # BatchNorm memory for the current weights
        val_loss, y_val, p_val = evaluate(model, val_loader, loss_fn, device)
        table = classification_metrics(y_val, p_val, threshold, categories)
        val_f1 = table.loc["macro", "f1"]
        val_auc = table.loc["macro", "auc"]
        seconds = time.time() - start
        print(
            f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"macro F1 {val_f1:.3f}  macro AUC {val_auc:.3f}  {seconds:.1f} s"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "val_macro_auc": val_auc,
                "seconds": seconds,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    return best_state, best_epoch, best_val_loss, history
