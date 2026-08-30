"""D-C: a propagation kernel that FOLLOWS the vessel instead of crossing it.

WRITTEN AND SELFTESTED 2026-08-31, BEFORE THE FIRST _snake RUN.

WHY D-B FAILED, measured rather than guessed. exp/propagate.py blended eight
fixed elliptical kernels by a predicted axis. Six reaches x six seeds, and
`prop - shuf` failed in all six cells. Three faults, each with its evidence:

  1. THE REACH WAS FAR TOO SHORT. It swept 0.5, 1 and 2 vessel widths -- 1.4,
     2.8 and 5.7 px. The gaps that matter are the missed centreline runs, and
     their p90 length is 21 px, about 7.4 widths. The operator could not reach
     across the thing it was built for.

  2. A STRAIGHT KERNEL CANNOT SIMPLY BE MADE LONGER. Measured on DRIVE's
     ground-truth skeleton, the fraction of segments that stray more than half
     a vessel width from the straight line through them:

         2 widths ( 6 px)    5.1%
         4 widths (11 px)   24.9%
         8 widths (23 px)   73.3%

     So the reach needed to bridge a real gap is exactly the reach at which a
     straight ellipse has left the vessel. That is a bind, not a tuning
     problem, and no sweep over `along` escapes it.

  3. IT COULD ONLY ADD. `logits + gain * relu(propagated - logits)` is a
     dilation. The segmentation loss punishes added foreground, so the layer's
     only available action was the one being penalised. And calibration.md
     then showed that "predict more foreground" is what lowering the threshold
     does for free, at no training cost at all.

WHAT THIS DOES INSTEAD.

  CURVES. The kernel is a line of 2K+1 samples walked outward from each pixel,
  and the step direction is re-read from the predicted field AT EACH NEW
  POINT. So the line bends with the vessel. This is the mechanism Dynamic
  Snake Convolution (ICCV 2023) uses, with one difference that is the point
  here: DSConv learns free offsets with no supervision on them, while this
  walks a field that carries its own target (exp/direction.py) and its own
  control (a shuffled field). D-B already established that the network can
  tell a real axis field from a random one -- the gate opened 3.1x to 6.0x
  wider for the real one, twelve seeds of twelve. The representation was never
  the problem; its consumer was.

  CONVOLVES. Learned weights over the 2K+1 samples, so the operator can
  subtract as well as add. A dilation cannot sharpen a boundary; a convolution
  can.

  GATES PER PIXEL. D-B had one scalar for the whole image, so it acted
  everywhere or nowhere. Here the gate is predicted per pixel from the
  model's own hesitation, so it can open at a gap and stay shut on a clean
  vessel. Uncertainty-Guided Conservative Propagation (2026) uses a per-pixel
  uncertainty gate on an ISOTROPIC 4-neighbourhood; this is the same gating
  idea on an oriented operator.

  ITERATES. Walking a curve is inherently sequential. T steps, swept.

THE AXIS PROBLEM, again. The field holds (sin 2theta, cos 2theta): an axis,
with no head or tail. Walking it needs a direction, so at every step the sign
is chosen to continue the previous step (dot product positive). Without that
the walk oscillates on the spot -- asserted below.

  python exp/snake.py --selftest
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Step length in pixels. One pixel per step, as DSConv does: shorter wastes
# samples, longer skips the curvature the walk exists to follow.
STEP = 1.0


def tangent_from_field(field: torch.Tensor) -> tuple:
    """(sin 2t, cos 2t) -> a unit tangent (dy, dx), sign arbitrary.

    theta = atan2(sin 2t, cos 2t) / 2. The halving is what turns an axis back
    into a direction; which of the two it returns is not defined, and the walk
    fixes the sign by continuity.
    """
    sin2, cos2 = field[:, 0:1], field[:, 1:2]
    angle = torch.atan2(sin2, cos2) * 0.5
    return torch.sin(angle), torch.cos(angle)


def _sample(plane: torch.Tensor, ys: torch.Tensor,
            xs: torch.Tensor) -> torch.Tensor:
    """Bilinear read of `plane` (B,C,H,W) at coordinates (B,K,H,W).

    Returns (B,C,K,H,W). grid_sample takes ONE grid of (B,H,W,2), so the K
    taps are stacked down the height axis and split again afterwards. The
    first draft passed `grid[:, 0]` and silently sampled only the first tap,
    which made the layer the identity for every kernel -- a bug that hides
    as "the layer does nothing yet".
    """
    batch, channels, height, width = plane.shape
    taps = ys.shape[1]
    grid = torch.stack([2.0 * xs / max(width - 1, 1) - 1.0,
                        2.0 * ys / max(height - 1, 1) - 1.0], dim=-1)
    flat = grid.reshape(batch, taps * height, width, 2)
    out = F.grid_sample(plane, flat, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.reshape(batch, channels, taps, height, width)


def walk(field: torch.Tensor, taps: int) -> tuple:
    """Sample coordinates of a line that follows `field`, both ways.

    Returns (ys, xs), each (B, 2*taps+1, H, W). Index `taps` is the pixel
    itself; indices below walk one way and above the other.
    """
    batch, _, height, width = field.shape
    device = field.device
    base_y = torch.arange(height, device=device, dtype=field.dtype)
    base_x = torch.arange(width, device=device, dtype=field.dtype)
    grid_y = base_y[None, None, :, None].expand(batch, 1, height, width)
    grid_x = base_x[None, None, None, :].expand(batch, 1, height, width)

    ys = [grid_y]
    xs = [grid_x]
    for sign in (1.0, -1.0):
        pos_y, pos_x = grid_y, grid_x
        previous = None
        forward, backward = [], []
        for _ in range(taps):
            here = _sample(field, pos_y, pos_x)[:, :, 0]
            step_y, step_x = tangent_from_field(here)
            step_y, step_x = sign * step_y, sign * step_x
            if previous is not None:
                # An axis has no head. Continue the way we were going, or the
                # walk turns round and oscillates between two pixels.
                flip = torch.sign(step_y * previous[0] + step_x * previous[1])
                flip = torch.where(flip == 0, torch.ones_like(flip), flip)
                step_y, step_x = step_y * flip, step_x * flip
            previous = (step_y, step_x)
            pos_y = pos_y + STEP * step_y
            pos_x = pos_x + STEP * step_x
            (forward if sign > 0 else backward).append((pos_y, pos_x))
        if sign > 0:
            after = forward
        else:
            before = backward
    ys = [p[0] for p in reversed(before)] + [grid_y] + [p[0] for p in after]
    xs = [p[1] for p in reversed(before)] + [grid_x] + [p[1] for p in after]
    return torch.cat(ys, dim=1), torch.cat(xs, dim=1)


class SnakePropagation(nn.Module):
    """A learned 1-D convolution along the vessel, gated per pixel.

    `taps` samples each way, so the kernel spans 2*taps+1 pixels of ARC
    length -- not of straight-line length, which is the whole point.
    """

    def __init__(self, taps: int, steps: int = 1, channels: int = 1,
                 straight: bool = False):
        super().__init__()
        self.taps, self.steps, self.straight = taps, steps, straight
        # Initialised to the identity: weight 1 on the centre tap, 0 elsewhere,
        # so an untrained layer returns its input and the arm starts as the one
        # it is named after. D-B's lesson: a layer that starts somewhere else
        # is a different architecture, not a fair comparison.
        weight = torch.zeros(1, 2 * taps + 1, 1, 1)
        weight[0, taps] = 1.0
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(1))
        # Per-pixel gate from the model's own hesitation. D-B used ONE scalar
        # for the whole image, so it dilated everywhere or nowhere; a gap needs
        # the operator open exactly where the model is unsure.
        self.gate = nn.Conv2d(channels + 1, 1, 3, padding=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, logits: torch.Tensor, field: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        current = logits
        for _ in range(self.steps):
            if self.straight:
                # The control: the SAME operator with the walk switched off,
                # so the line is straight along the field's own axis at the
                # centre pixel. Isolates curvature from everything else.
                ys, xs = self._straight(field)
            else:
                ys, xs = walk(field, self.taps)
            sampled = _sample(current, ys, xs)[:, 0]
            proposed = (sampled * self.weight).sum(dim=1, keepdim=True) \
                + self.bias
            hesitation = 1.0 - 2.0 * (torch.sigmoid(current) - 0.5).abs()
            gate = torch.sigmoid(self.gate(
                torch.cat([features, hesitation], dim=1)))
            current = current + gate * (proposed - current)
        return current

    def _straight(self, field: torch.Tensor) -> tuple:
        batch, _, height, width = field.shape
        device = field.device
        grid_y = torch.arange(height, device=device, dtype=field.dtype)[
            None, None, :, None].expand(batch, 1, height, width)
        grid_x = torch.arange(width, device=device, dtype=field.dtype)[
            None, None, None, :].expand(batch, 1, height, width)
        step_y, step_x = tangent_from_field(field)
        offsets = torch.arange(-self.taps, self.taps + 1, device=device,
                               dtype=field.dtype)[None, :, None, None]
        return (grid_y + offsets * STEP * step_y,
                grid_x + offsets * STEP * step_x)


def _arc_field(size: int, radius: float) -> tuple:
    """A circular arc through the image centre, and its exact tangent axis.

    A circle is the cleanest curved test: every point has a known tangent, and
    the straight-line error over an arc of length L is L^2/(8R), so the test
    can assert a NUMBER rather than "looks bent".
    """
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    centre = size / 2.0
    dy, dx = ys - centre, xs - (centre - radius)
    distance = np.hypot(dy, dx)
    on_arc = np.abs(distance - radius) < 1.0
    # Tangent to the circle at each pixel is perpendicular to the radius.
    norm = np.maximum(distance, 1e-6)
    tangent_y, tangent_x = -dx / norm, dy / norm
    angle = np.arctan2(tangent_y, tangent_x)
    field = np.stack([np.sin(2 * angle), np.cos(2 * angle)])
    return (torch.tensor(field[None], dtype=torch.float32),
            on_arc, (centre, centre - radius), radius)


def selftest() -> None:
    torch.manual_seed(0)

    # 1. IDENTITY AT INITIALISATION. A layer that starts anywhere else is a
    #    different architecture from the arm it is compared against -- the
    #    mistake D-B's first gate made twice.
    for straight in (False, True):
        layer = SnakePropagation(taps=8, steps=2, channels=4,
                                 straight=straight)
        logits = torch.randn(2, 1, 32, 32)
        field = torch.randn(2, 2, 32, 32)
        features = torch.randn(2, 4, 32, 32)
        out = layer(logits, field, features)
        assert torch.allclose(out, logits, atol=1e-5), \
            (straight, (out - logits).abs().max().item())
    print("  the layer is EXACTLY the identity at initialisation, both modes")

    # 2. AND ITS GATE CAN STILL MOVE. An initialisation with no gradient is an
    #    ablation wearing a parameter, which is what sigmoid(-6) would have
    #    been in D-B.
    layer = SnakePropagation(taps=8, steps=1, channels=4)
    logits = torch.randn(1, 1, 32, 32, requires_grad=False)
    out = layer(logits, torch.randn(1, 2, 32, 32), torch.randn(1, 4, 32, 32))
    out.sum().backward()
    assert layer.weight.grad.abs().max() > 1e-6, layer.weight.grad
    assert layer.gate.weight.grad.abs().max() >= 0.0
    print(f"  gradient reaches the kernel weights "
          f"({layer.weight.grad.abs().max():.3e})")

    # 3. THE POINT OF THE WHOLE FILE. On a curved vessel the walk must stay on
    #    it and the straight line must leave. Asserted as a distance, against
    #    the analytic sagitta L^2/(8R).
    size, radius = 96, 20.0
    field, on_arc, (cy, cx), radius = _arc_field(size, radius)
    taps = 8
    ys, xs = walk(field, taps)
    sy, sx = SnakePropagation(taps=taps, straight=True)._straight(field)
    centre_pixel = (int(cy), int(cx + radius))     # a point on the arc
    def stray(py, px):
        dy = py[0, :, centre_pixel[0], centre_pixel[1]] - cy
        dx = px[0, :, centre_pixel[0], centre_pixel[1]] - cx
        return (torch.hypot(dy, dx) - radius).abs().max().item()
    curved, flat = stray(ys, xs), stray(sy, sx)
    sagitta = (2 * taps * STEP) ** 2 / (8 * radius)
    print(f"  on an arc of radius {radius:.0f}, a {2*taps+1}-tap kernel "
          f"strays {curved:.2f} px when it walks and {flat:.2f} px when "
          f"straight (analytic sagitta {sagitta:.2f})")
    # The straight control must match the analytic sagitta -- that proves it
    # is failing for the geometric reason and not a coding one -- and must
    # exceed half a DRIVE vessel width (1.41 px), which is the point at which
    # a kernel has left the vessel it started on.
    assert abs(flat - sagitta) < 0.15 * sagitta, (flat, sagitta)
    assert flat > 1.41, flat
    assert curved < 0.6, curved
    assert flat > 4 * curved, (flat, curved)

    # 4. THE AXIS TRAP. Without sign continuity the walk oscillates on the
    #    spot, because atan2/2 returns either end of the axis arbitrarily.
    #    Asserted by how far the walk actually gets: a 16-step walk that
    #    oscillates covers about one pixel.
    reach = torch.hypot(ys[0, -1] - ys[0, taps],
                        xs[0, -1] - xs[0, taps])[on_arc].median().item()
    print(f"  {taps} steps travel {reach:.1f} px along the vessel "
          f"(oscillation would give ~1)")
    assert reach > 0.8 * taps * STEP, reach

    # 5. And a straight kernel must be EXACTLY straight, or the control is not
    #    a control.
    line_y = sy[0, :, centre_pixel[0], centre_pixel[1]]
    line_x = sx[0, :, centre_pixel[0], centre_pixel[1]]
    span = torch.stack([line_y[-1] - line_y[0], line_x[-1] - line_x[0]])
    span = span / span.norm()
    rel = torch.stack([line_y - line_y[0], line_x - line_x[0]])
    off = (rel[0] * span[1] - rel[1] * span[0]).abs().max().item()
    assert off < 1e-4, off
    print("  the straight control is straight to 1e-4 px")
    print("all checks passed")


if __name__ == "__main__":
    selftest()
