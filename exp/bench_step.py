"""Measure what a training step actually costs on this GPU, per config.

Every hour figure in prompt.md section 4 was extrapolated from CPU timings on
a laptop. Extrapolation across a device change is a guess, and a queue budgeted
from a guess is how a night gets spent on the wrong three runs. Five minutes
here replaces the whole table.

Two numbers per config, because a run pays both:
  step     one forward+backward at the real BATCH and PATCH, after warm-up
  val      one whole-image pass over the validation split, which happens
           EVERY_VAL epochs and at 31M parameters is no longer free

torch.cuda.synchronize() around both is not optional: CUDA launches are
asynchronous, so timing without it measures how fast Python queued the work.
That mistake reports a 20x speedup that is not there.

  python exp/bench_step.py                       # the task-1 configs
  python exp/bench_step.py A_dice B_cldice_w32   # any names from CONFIGS
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

DEFAULT = ("A_dice", "A_dice_w32", "A_dice_w64_d5", "B_cldice_w64_d5",
           "G_focal_w64_d5", "K_focal_aug_w64_d5")
WARMUP, REPEATS = 8, 40


def time_steps(config_name: str, data: dict, mean: float, std: float) -> float:
    """Milliseconds per optimiser step, including the data pipeline.

    The batch is resampled every step rather than reused: sample_batch is 1-5 ms
    of numpy per step and a queue budget that leaves it out is short by that
    much times 31,200. Measuring the real loop is cheaper than arguing about
    whether the omission matters.
    """
    _, extra = train.CONFIGS[config_name]
    augments = train.AUGMENTS.get(config_name, ())
    model = train.build_model(config_name)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = __import__("numpy").random.default_rng(0)

    for index in range(WARMUP + REPEATS):
        if index == WARMUP:                      # cuDNN autotune is done
            torch.cuda.synchronize()
            started = time.time()
        images, labels, dists = train.sample_batch(
            data, rng, mean, std, augments)
        optimiser.zero_grad()
        train.compute_loss(model(images), labels, dists, extra,
                           images).backward()
        optimiser.step()
    torch.cuda.synchronize()
    return 1000 * (time.time() - started) / REPEATS


def time_validation(config_name: str, val: dict, mean: float,
                    std: float) -> float:
    """Seconds for one validation pass, model forward only.

    metrics.evaluate is deliberately excluded: Betti numbers and 95HD are
    scipy on the CPU and cost the same whatever the GPU does, so mixing them in
    would hide the number this function exists to expose.
    """
    model = train.build_model(config_name)
    train.predict_full(model, val["images"][0], mean, std)   # warm up
    torch.cuda.synchronize()
    started = time.time()
    for index in range(len(val["names"])):
        train.predict_full(model, val["images"][index], mean, std)
    torch.cuda.synchronize()
    return time.time() - started


def main() -> None:
    names = sys.argv[1:] or list(DEFAULT)
    print(f"device {train.DEVICE}", flush=True)
    if train.DEVICE.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    data, val = train.stack_split("train"), train.stack_split("val")
    inside = data["images"][data["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())
    steps = train.PATCHES_PER_EPOCH // train.BATCH
    validations = train.EPOCHS // train.VAL_EVERY

    print(f"\n{train.EPOCHS} epochs x {steps} steps = "
          f"{train.EPOCHS * steps:,} steps at batch {train.BATCH}, "
          f"patch {train.PATCH}\n")
    header = (f"{'config':<20} {'params':>12} {'ms/step':>9} {'val s':>7} "
              f"{'train h':>8} {'val h':>7} {'run h':>7}")
    print(header)
    print("-" * len(header))
    for name in names:
        params = sum(p.numel() for p in train.build_model(name).parameters())
        ms = time_steps(name, data, mean, std)
        val_s = time_validation(name, val, mean, std)
        train_h = ms * steps * train.EPOCHS / 3_600_000
        val_h = val_s * validations / 3600
        print(f"{name:<20} {params:>12,} {ms:>9.1f} {val_s:>7.1f} "
              f"{train_h:>8.2f} {val_h:>7.2f} {train_h + val_h:>7.2f}",
              flush=True)


if __name__ == "__main__":
    main()
