#!/usr/bin/env python3
"""Exact descriptor contract for the three local reference artifacts.

Shapes follow cd959bae99ff4bed22bef2604f45241c09bbb8c8 model loaders.
This is an audit contract, not a runtime or a general Gemma loader.
"""


def expected(role):
    result = {}

    def add(name, shape, kind='F32'):
        result[name] = (shape, kind)

    if role in ('target', 'assistant'):
        target = role == 'target'
        emb, layers = (2816, 30) if target else (1024, 4)
        add('token_embd.weight', [emb, 262144], 'Q4_0')
        add('output_norm.weight', [emb])
        add('rope_freqs.weight', [256])
        if not target:
            add('nextn.pre_projection.weight', [5632, 1024], 'Q4_0')
            add('nextn.post_projection.weight', [1024, 2816], 'Q4_0')
        for layer in range(layers):
            local = layer % 6 != 5 if target else layer < 3
            head = 256 if local else 512
            prefix = f'blk.{layer}.'
            norms = ['attn_norm', 'ffn_norm', 'post_attention_norm', 'post_ffw_norm']
            if target:
                norms += ['post_ffw_norm_1', 'post_ffw_norm_2', 'pre_ffw_norm_2']
            for name in norms:
                add(prefix + name + '.weight', [emb])
            add(prefix + 'layer_output_scale.weight', [1])
            add(prefix + 'attn_q_norm.weight', [head])
            add(prefix + 'attn_q.weight', [emb, head * 16], 'Q4_0')
            add(prefix + 'attn_output.weight', [head * 16, emb], 'Q4_0')
            ff = 2112 if target else 8192
            for name in ['ffn_gate', 'ffn_up']:
                add(prefix + name + '.weight', [emb, ff], 'Q4_0')
            add(prefix + 'ffn_down.weight', [ff, emb], 'Q4_0')
            if target:
                kv = head * (8 if local else 2)
                add(prefix + 'attn_k.weight', [emb, kv], 'Q4_0')
                add(prefix + 'attn_k_norm.weight', [head])
                if local:
                    add(prefix + 'attn_v.weight', [emb, kv], 'Q4_0')
                add(prefix + 'ffn_gate_inp.weight', [emb, 128])
                add(prefix + 'ffn_gate_inp.scale', [emb])
                add(prefix + 'ffn_down_exps.scale', [128])
                add(prefix + 'ffn_gate_up_exps.weight', [emb, 1408, 128], 'Q4_0')
                add(prefix + 'ffn_down_exps.weight', [704, emb, 128], 'Q4_0')
    elif role == 'projector':
        add('mm.input_projection.weight', [1152, 2816])
        add('v.patch_embd.weight', [16, 16, 3, 1152])
        add('v.position_embd.weight', [1152, 10240, 2])
        add('v.std_bias', [1152])
        add('v.std_scale', [1152])
        for layer in range(27):
            prefix = f'v.blk.{layer}.'
            for name in ['ln1', 'ln2', 'attn_post_norm', 'ffn_post_norm']:
                add(prefix + name + '.weight', [1152])
            for name in ['attn_q_norm', 'attn_k_norm']:
                add(prefix + name + '.weight', [72])
            for name in ['attn_q', 'attn_k', 'attn_v', 'attn_out']:
                add(prefix + name + '.weight', [1152, 1152])
            for name in ['ffn_gate', 'ffn_up']:
                add(prefix + name + '.weight', [1152, 4304])
            add(prefix + 'ffn_down.weight', [4304, 1152])
    else:
        raise ValueError('unknown artifact role')
    return result


def validate(role, tensors):
    """Reject missing, extra, duplicate, wrong-shaped or wrong-typed descriptors."""
    contract = expected(role)
    seen = set()
    for tensor in tensors:
        name = tensor['name']
        if name in seen or contract.get(name) != (tensor['shape'], tensor['type']):
            raise ValueError('descriptor contract mismatch')
        seen.add(name)
    if seen != contract.keys():
        raise ValueError('incomplete descriptor contract')
    return len(seen)
