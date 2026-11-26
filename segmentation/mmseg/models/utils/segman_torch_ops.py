"""PyTorch implementations of the SegMAN operators used on Windows."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def cross_scan(x):
    row = x.flatten(-2)
    column = x.transpose(-2, -1).flatten(-2)
    return torch.stack((row, column, row.flip(-1), column.flip(-1)), dim=1)


def cross_merge(scans):
    height, width = scans.shape[-2:]
    row, column, reverse_row, reverse_column = scans.flatten(-2).unbind(1)
    column = column.unflatten(-1, (width, height)).transpose(-2, -1)
    reverse_column = reverse_column.flip(-1).unflatten(-1, (width, height))
    reverse_column = reverse_column.transpose(-2, -1)
    return (row + column.flatten(-2) + reverse_row.flip(-1) +
            reverse_column.flatten(-2))


def _scan_recurrence(decay, increment):
    """Evaluate s[t] = decay[t] * s[t - 1] + increment[t] in parallel."""
    length = increment.shape[-1]
    if length == 1:
        return increment

    # Compose adjacent steps, recursively solve the odd positions, then fill
    # each even position from its immediate predecessor.
    n_pairs = length // 2
    odd_decay = decay[..., 1::2]
    odd_increment = increment[..., 1::2]
    paired_decay = odd_decay * decay[..., :2 * n_pairs:2]
    paired_increment = odd_increment + odd_decay * increment[..., :2 * n_pairs:2]
    odd_states = _scan_recurrence(paired_decay, paired_increment)
    even_states = increment[..., 2::2] + decay[..., 2::2] * odd_states[
        ..., :(length - 1) // 2]
    even_states = torch.cat((increment[..., :1], even_states), dim=-1)
    paired_states = torch.stack(
        (even_states[..., :n_pairs], odd_states), dim=-1).flatten(-2)
    return torch.cat((paired_states, even_states[..., n_pairs:]), dim=-1)


def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex):
    """Selective Scan with real weights and per-group B and C."""
    batch, groups, states, length = B.shape
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[:, None]
    if delta_softplus:
        delta = F.softplus(delta)
    u_float = u.float()

    decay = (delta.unsqueeze(-2) * A[None, :, :, None]).exp()
    decay = decay.reshape(batch, groups, -1, states, length)
    drive = (delta * u_float).reshape(batch, groups, -1, 1, length)
    hidden = _scan_recurrence(decay, drive * B.float().unsqueeze(2))
    output = (hidden * C.float().unsqueeze(2)).sum(dim=-2).reshape_as(u)
    if D is not None:
        output = output + u_float * D[:, None]
    return output if oflex else output.to(u.dtype)


def _axis_neighbors(positions, length, kernel, dilation):
    phase = positions % dilation
    center = positions // dilation
    group_len = (length - 1 - phase) // dilation + 1
    start = (center - kernel // 2).clamp_min(0).minimum(group_len - kernel)
    steps = start[:, None] + torch.arange(kernel, device=positions.device)
    return steps * dilation + phase[:, None], steps - center[:, None] + kernel - 1


def _attend(query, key, value, indices, bias, scale):
    batch, heads, queries, channels = query.shape
    shape = (batch, heads, queries, -1, channels)
    key = key.index_select(2, indices).reshape(shape)
    value = value.index_select(2, indices).reshape(shape)
    logits = (query.unsqueeze(-2) @ key.transpose(-2, -1)) * scale
    if bias is not None:
        logits = logits + bias.to(logits.dtype)
    return (logits.softmax(dim=-1) @ value).squeeze(-2)


def neighborhood_attention(query, key, value, kernel_size, dilation,
                           rpb=None, scale=1.0):
    """Sliding windows with shifted boundaries on B x heads x H x W x C."""
    height, width = query.shape[2:4]
    kh, kw = kernel_size
    dh, dw = dilation
    if kh * dh > height or kw * dw > width:
        raise ValueError('The dilated attention window exceeds the feature map')

    query_flat, key_flat, value_flat = (
        tensor.flatten(2, 3) for tensor in (query, key, value))
    outputs = []
    # Limit materialized neighborhoods and recompute them during backward.
    for start in range(0, height * width, 256):
        positions = torch.arange(
            start, min(start + 256, height * width), device=query.device)
        rows, row_bias = _axis_neighbors(positions // width, height, kh, dh)
        cols, col_bias = _axis_neighbors(positions % width, width, kw, dw)
        indices = (rows[:, :, None] * width + cols[:, None, :]).flatten()
        bias = None
        if rpb is not None:
            bias = rpb[:, row_bias[:, :, None], col_bias[:, None, :]]
            bias = bias.flatten(-2).unsqueeze(0).unsqueeze(-2)
        args = (query_flat[:, :, start:start + positions.numel()],
                key_flat, value_flat, indices, bias, scale)
        if torch.is_grad_enabled():
            outputs.append(checkpoint(_attend, *args, use_reentrant=False,
                                      preserve_rng_state=False))
        else:
            outputs.append(_attend(*args))
    return torch.cat(outputs, dim=2).reshape_as(query)
