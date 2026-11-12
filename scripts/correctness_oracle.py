#!/usr/bin/env python3
"""Independent scalar reference math, not a captured model correctness oracle.

Python float reductions are binary64. Q4_0 decodes little-endian fp16 scale,
low nibbles first (positions 0..15), then high nibbles (16..31), offset by 8.
No quantizer, graph implementation or llama.cpp helper is shared.
"""
import math
import struct
import ctypes


def q4_0_decode(block):
    if len(block) != 18:
        raise ValueError('Q4_0 requires exactly 18 bytes')
    scale = struct.unpack('<e', block[:2])[0]
    if not math.isfinite(scale):
        raise ValueError('nonfinite Q4_0 scale')
    return [scale * ((value & 15) - 8) for value in block[2:]] + [
        scale * ((value >> 4) - 8) for value in block[2:]]


def finite_vector(values):
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError('nonempty finite vector required')


def embedding_lookup(table, token_ids, scale):
    """Gather explicit embedding rows and apply an explicit scalar scale."""
    if not isinstance(table, list) or not table or not isinstance(token_ids, list) or not token_ids:
        raise ValueError('nonempty embedding table and token IDs required')
    width = len(table[0]) if isinstance(table[0], list) else 0
    if width == 0:
        raise ValueError('embedding rows must not be empty')
    for row in table:
        finite_vector(row)
        if len(row) != width:
            raise ValueError('embedding table shape mismatch')
    if (any(type(token_id) is not int or token_id < 0 or token_id >= len(table)
            for token_id in token_ids)
            or type(scale) not in (int, float) or not math.isfinite(scale)):
        raise ValueError('invalid embedding token ID or finite scale')
    result = [[value * scale for value in table[token_id]] for token_id in token_ids]
    for row in result:
        finite_vector(row)
    return result


def matvec(rows, vector):
    finite_vector(vector)
    if not rows:
        raise ValueError('matrix must not be empty')
    result = []
    for row in rows:
        finite_vector(row)
        if len(row) != len(vector):
            raise ValueError('matrix/vector shape mismatch')
        result.append(math.fsum(a * b for a, b in zip(row, vector)))
    finite_vector(result)
    return result


def vector_add(left, right):
    """Independent binary64 elementwise residual/addition reference."""
    finite_vector(left)
    finite_vector(right)
    if len(left) != len(right):
        raise ValueError('vector addition shape mismatch')
    result = [math.fsum((a, b)) for a, b in zip(left, right)]
    finite_vector(result)
    return result


def weighted_reduce(vectors, weights):
    """Reduce already-selected vectors with explicit, unnormalized weights."""
    if not isinstance(vectors, list) or not vectors or not isinstance(weights, list):
        raise ValueError('nonempty vectors and explicit weights required')
    if len(vectors) != len(weights):
        raise ValueError('vector/weight count mismatch')
    width = len(vectors[0])
    if width == 0:
        raise ValueError('reduction vectors must not be empty')
    for vector in vectors:
        finite_vector(vector)
        if len(vector) != width:
            raise ValueError('weighted reduction shape mismatch')
    if any(type(weight) not in (int, float) or not math.isfinite(weight) for weight in weights):
        raise ValueError('finite explicit reduction weights required')
    result = [math.fsum(weight * vector[column]
                        for vector, weight in zip(vectors, weights))
              for column in range(width)]
    finite_vector(result)
    return result


def tensor_errors(reference, candidate):
    finite_vector(reference)
    finite_vector(candidate)
    if len(reference) != len(candidate):
        raise ValueError('tensor shape mismatch')
    # Scale before squaring to avoid overflow for finite large inputs.
    scale = max(max(map(abs, reference)), max(map(abs, candidate)), 1e-300)
    ref = [r / scale for r in reference]
    cand = [c / scale for c in candidate]
    errors = [c - r for r, c in zip(ref, cand)]
    rr = math.fsum(r * r for r in ref)
    cc = math.fsum(c * c for c in cand)
    rmse = math.sqrt(math.fsum(e * e for e in errors) / len(ref))
    ref_rms = math.sqrt(rr / len(ref))
    return {'normalized_rmse': rmse / max(ref_rms, 1e-12 / scale),
            'cosine': math.fsum(r * c for r, c in zip(ref, cand)) / math.sqrt(rr * cc)
            if rr and cc else None,
            'max_absolute_error': max(abs(c - r) for r, c in zip(reference, candidate)),
            'max_relative_error': max(abs(c - r) / max(abs(r), 1e-12)
                                      for r, c in zip(reference, candidate))}


def log_softmax(logits):
    finite_vector(logits)
    peak = max(logits)
    shifted = [x - peak for x in logits]
    normalizer = math.log(math.fsum(math.exp(x) for x in shifted))
    result = [x - normalizer for x in shifted]
    finite_vector(result)
    return result


def logit_errors(reference_rows, candidate_rows):
    if not reference_rows or len(reference_rows) != len(candidate_rows):
        raise ValueError('matched logit positions required')
    kls, matches, errors = [], 0, []
    width = len(reference_rows[0])
    for reference, candidate in zip(reference_rows, candidate_rows):
        if len(reference) != width or len(candidate) != width:
            raise ValueError('full vocabulary width must match at every position')
        errors.append(tensor_errors(reference, candidate))
        rp, cp = log_softmax(reference), log_softmax(candidate)
        kl = math.fsum(math.exp(r) * (r - c) for r, c in zip(rp, cp))
        if kl < -1e-12:
            raise ValueError('negative KL beyond rounding envelope')
        kls.append(max(0.0, kl))
        matches += max(range(width), key=reference.__getitem__) == max(range(width), key=candidate.__getitem__)
    return {'positions': len(kls), 'vocabulary': width,
            'mean_kl_nats': math.fsum(kls) / len(kls),
            'top1_agreement': matches / len(kls), 'per_position_errors': errors,
            'sufficient_positions': len(kls) >= 4096}


def router_margin(logits, selected, k):
    finite_vector(logits)
    if type(k) is not int or not 0 < k < len(logits):
        raise ValueError('router k must leave an unselected expert')
    ranked = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    if len(selected) != k or len(set(selected)) != k or any(type(i) is not int or i < 0 or i >= len(logits) for i in selected):
        raise ValueError('invalid selected expert IDs')
    return {'reference_ids': ranked[:k], 'selected_set_matches': set(selected) == set(ranked[:k]),
            'boundary_margin': logits[ranked[k - 1]] - logits[ranked[k]],
            'tie_policy': 'ascending expert ID for this scalar diagnostic only'}


def rms_norm(vector, epsilon, weights=None):
    """Binary64 RMS reference for small finite fixtures, optionally weighted."""
    finite_vector(vector)
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('positive finite normalization epsilon required')
    if weights is not None:
        finite_vector(weights)
        if len(weights) != len(vector):
            raise ValueError('normalization weight shape mismatch')
    mean_square = math.fsum(x * x for x in vector) / len(vector)
    if not math.isfinite(mean_square):
        raise ValueError('normalization fixture exceeds binary64 range')
    inverse = 1 / math.sqrt(mean_square + epsilon)
    result = [x * inverse * (weights[i] if weights is not None else 1)
              for i, x in enumerate(vector)]
    finite_vector(result)
    return result


def router_projection(vector, epsilon, inverse_sqrt_width, learned_input_scale,
                      projection_rows):
    """Scalar Gemma 4 router chain with every scale supplied explicitly.

    Mirrors the pinned source order: RMSNorm(attn_out), multiply by
    1/sqrt(n_embd), elementwise learned input scale, then router projection.
    This does not infer model tensor shapes or emulate FP32 rounding.
    """
    finite_vector(vector)
    if (type(inverse_sqrt_width) not in (int, float)
            or not math.isfinite(inverse_sqrt_width) or inverse_sqrt_width <= 0):
        raise ValueError('positive finite inverse-square-root width required')
    finite_vector(learned_input_scale)
    if len(learned_input_scale) != len(vector):
        raise ValueError('router learned input scale shape mismatch')
    normalized = rms_norm(vector, epsilon)
    width_scaled = [value * inverse_sqrt_width for value in normalized]
    finite_vector(width_scaled)
    input_scaled = [value * scale for value, scale
                    in zip(width_scaled, learned_input_scale)]
    finite_vector(input_scaled)
    return {'rms_norm': normalized, 'width_scaled': width_scaled,
            'input_scaled': input_scaled, 'logits': matvec(projection_rows, input_scaled)}


def gelu_tanh(vector):
    """Tanh-approximate GELU, not the erf variant or an FP32 rounding emulator."""
    finite_vector(vector)
    result = [.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * x * (1 + .044715 * x * x)))
              for x in vector]
    finite_vector(result)
    return result


def rope_rotate(vector, pairs, angles):
    """Apply explicit independent 2-D rotations to disjoint vector pairs.

    Position/frequency derivation is deliberately outside this helper; callers
    must bind those inputs and the pair layout to the reference configuration.
    Unpaired dimensions pass through unchanged.
    """
    finite_vector(vector)
    if not isinstance(pairs, list) or not isinstance(angles, list) or len(pairs) != len(angles):
        raise ValueError('matched explicit rotary pairs and angles required')
    result, used = list(vector), set()
    for pair, angle in zip(pairs, angles):
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(index) is not int or index < 0 or index >= len(vector) for index in pair)
                or pair[0] == pair[1] or pair[0] in used or pair[1] in used
                or type(angle) not in (int, float) or not math.isfinite(angle)):
            raise ValueError('invalid, overlapping or nonfinite rotary pair')
        left, right = pair
        cosine, sine = math.cos(angle), math.sin(angle)
        x, y = vector[left], vector[right]
        result[left], result[right] = x * cosine - y * sine, x * sine + y * cosine
        used.update(pair)
    finite_vector(result)
    return result


def softcap(vector, cap):
    """Independent scalar `cap * tanh(x / cap)` reference for final logits."""
    finite_vector(vector)
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
        raise ValueError('positive finite softcap required')
    result = [cap * math.tanh(value / cap) for value in vector]
    finite_vector(result)
    return result


def gelu_cpu_fp16_lookup(vector):
    """Model pinned ggml CPU's FP32 GELU path with its FP16 lookup semantics.

    This is a precision-specific diagnostic reference, not a replacement for
    ``gelu_tanh`` or its mathematical-FP32 acceptance criterion. Finite inputs
    are rounded to FP32, values at/beyond +/-10 take the source-defined shortcut,
    and interior values are rounded through FP16 input and output. ``tanhf`` is
    called through the platform C math library to preserve the source's float
    operation rather than Python's double-precision ``math.tanh``.
    """
    finite_vector(vector)
    try:
        libm = ctypes.CDLL('libm.so.6')
        tanhf = libm.tanhf
        tanhf.argtypes = [ctypes.c_float]
        tanhf.restype = ctypes.c_float
        fmaf = libm.fmaf
        fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        fmaf.restype = ctypes.c_float
    except (OSError, AttributeError) as exc:
        raise RuntimeError('platform tanhf is required for the CPU diagnostic') from exc

    def f32(value):
        try:
            return struct.unpack('<f', struct.pack('<f', value))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError('GELU input is outside finite FP32 range') from exc

    def f16(value):
        try:
            return struct.unpack('<e', struct.pack('<e', value))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError('GELU lookup value is outside finite FP16 range') from exc

    def source_gelu_f32(value):
        # Keep each operation rounded to float as in ggml's tanh approximation.
        # The pinned optimized CPU build contracts 1 + (0.044715*x)*x into
        # fmaf for this table initialization expression.
        inner = f32(fmaf(f32(f32(0.044715) * value), value, 1.0))
        scaled_x = f32(f32(0.7978845608028654) * value)
        argument = f32(scaled_x * inner)
        tanh_value = f32(tanhf(argument))
        return f32(f32(f32(0.5) * value) * f32(1.0 + tanh_value))

    result = []
    for original in vector:
        x = f32(original)
        if x <= -10.0:
            result.append(0.0)
        elif x >= 10.0:
            result.append(x)
        else:
            rounded_input = f16(x)
            result.append(f16(source_gelu_f32(rounded_input)))
    finite_vector(result)
    return result


def softmax(vector):
    return [math.exp(value) for value in log_softmax(vector)]


def router_selected_weights(logits, selected_ids,
                            normalization_floor=6.103515625e-5):
    """Check Gemma 4 softmax router weights for explicitly selected experts.

    Selection IDs are supplied by the reference because this helper does not
    emulate ggml argsort tie behavior. The selected probabilities are
    renormalized with the pinned graph's minimum denominator clamp.
    """
    finite_vector(logits)
    if (not isinstance(selected_ids, list) or not selected_ids
            or any(type(index) is not int or index < 0 or index >= len(logits)
                   for index in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)):
        raise ValueError('valid unique selected router IDs required')
    if (type(normalization_floor) not in (int, float)
            or not math.isfinite(normalization_floor) or normalization_floor <= 0):
        raise ValueError('positive finite router normalization floor required')
    probabilities = softmax(logits)
    selected = [probabilities[index] for index in selected_ids]
    total = math.fsum(selected)
    denominator = max(total, normalization_floor)
    weights = [weight / denominator for weight in selected]
    finite_vector(weights)
    return {'probabilities': probabilities, 'selected_ids': list(selected_ids),
            'selected_probabilities': selected, 'selected_sum': total,
            'normalization_denominator': denominator, 'weights': weights}


def gemma4_router_mixture(vector, epsilon, inverse_sqrt_width,
                          learned_input_scale, projection_rows, selected_ids,
                          selected_expert_outputs):
    """Compose scalar Gemma 4 router math and explicit selected-expert outputs.

    Expert IDs and corresponding output vectors are inputs in the same order;
    this checks the numeric mixture chain without emulating argsort, expert
    kernels, or producing any model-derived values.
    """
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError('explicit selected expert IDs required')
    if (not isinstance(selected_expert_outputs, list)
            or len(selected_expert_outputs) != len(selected_ids)):
        raise ValueError('one explicitly ordered output vector per selected expert required')
    router = router_projection(vector, epsilon, inverse_sqrt_width,
                              learned_input_scale, projection_rows)
    weights = router_selected_weights(router['logits'], selected_ids)
    mixture = weighted_reduce(selected_expert_outputs, weights['weights'])
    return {'router': router, 'selection_weights': weights, 'mixture': mixture}


def attention(query, keys, values, allowed, scale):
    """One-head scalar attention with explicit allowed positions and score scale.

    Inputs are already position-transformed. This reference does not implement
    KV caching, RoPE, grouped heads, flash kernels or local/global mask creation.
    All-masked rows fail; masked weights are exactly zero.
    """
    finite_vector(query)
    if not keys or len(keys) != len(values) or len(keys) != len(allowed) or any(type(v) is not bool for v in allowed) or not any(allowed):
        raise ValueError('matched nonempty KV/mask with a visible position required')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite attention scale required')
    width = len(values[0])
    for key, value in zip(keys, values):
        finite_vector(key)
        finite_vector(value)
        if len(key) != len(query) or len(value) != width:
            raise ValueError('attention shape mismatch')
    scores = [scale * math.fsum(q * k for q, k in zip(query, key))
              for key, visible in zip(keys, allowed) if visible]
    visible_weights = iter(softmax(scores))
    weights = [next(visible_weights) if visible else 0.0 for visible in allowed]
    output = [math.fsum(weight * value[column] for weight, value in zip(weights, values))
              for column in range(width)]
    finite_vector(output)
    return {'weights': weights, 'output': output}


def gated_ffn(vector, up_rows, gate_rows, down_rows):
    """Small GELU-gated FFN with explicit matrix rows; no norms/residuals."""
    up, gate = matvec(up_rows, vector), matvec(gate_rows, vector)
    if len(up) != len(gate):
        raise ValueError('gate/up width mismatch')
    activation = gelu_tanh(gate)
    product = [a * b for a, b in zip(activation, up)]
    return {'up': up, 'gate': gate, 'gated': product, 'output': matvec(down_rows, product)}
