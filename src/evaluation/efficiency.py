"""Step 1.5: the two satisficing metrics of an encoder, model size and CPU latency per image.
Neither depends on the weight values, only on the architecture."""

import copy
import io
import statistics
import time

import torch


def model_size_mb(model):
    """Size of the model's saved weights (its state_dict, as torch.save writes it) in MB,
    1 MB = 1,000,000 bytes. Saved to memory, not to disk."""
    buffer = io.BytesIO()  # a "file" that lives in RAM
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6


def cpu_latency_ms(model, image_size, threads, warmup, runs):
    """Median time in ms of one forward pass of a single image (batch 1, 1 channel) on the CPU,
    with `threads` CPU threads; the first `warmup` runs are not counted. The caller's model is
    not moved (a copy goes to the CPU)."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = copy.deepcopy(model).to("cpu")
    model.eval()  # inference: BatchNorm uses its running statistics
    x = torch.rand(
        1, 1, image_size, image_size
    )  # pixel values don't change the amount of work
    times = []
    with torch.no_grad():
        for i in range(warmup + runs):
            start = time.perf_counter()
            model(x)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if i >= warmup:  # skip the warm-up runs
                times.append(elapsed_ms)
    torch.set_num_threads(old_threads)  # leave PyTorch as we found it
    return statistics.median(times)
