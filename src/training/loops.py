"""Training and evaluation loops, shared by every PyTorch model (SmallCNN now, ResNet/MobileNet later)."""

import torch


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
