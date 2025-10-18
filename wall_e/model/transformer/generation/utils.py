import torch


def gather_beam_states(states, beam_indices, batch_size, num_beams):
    """
    Gather past_key_values for selected beams.
    states: list of tuples (key, value) per layer, each tensor has shape (batch*num_beams, ...)
    beam_indices: (batch, num_beams)
    """
    gather_indices = (
        beam_indices + (torch.arange(batch_size, device=beam_indices.device) * num_beams).unsqueeze(1)
    ).view(-1)

    new_states = []
    for layer_state in states:
        new_layer_state = tuple(x.index_select(0, gather_indices) for x in layer_state)
        new_states.append(new_layer_state)
    return new_states


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = torch.where(mask, torch.full_like(logits, filter_value), logits)
    return logits

