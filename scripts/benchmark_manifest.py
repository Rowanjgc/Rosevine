#!/usr/bin/env python3
"""Private run-identity schema for the proposed experiment protocol.

This schema inventories reproducibility evidence, not authorization. Null always
has a reason. A structurally complete manifest can still describe an invalid
experiment; human-reviewed protocol/run limits are separate requirements.
"""
import re

from benchmark_core import canonical, digest

SCHEMA = 'rosevine-private-run-v1'
# Every immutable-run-identity item in experiment-protocol.md is represented.
GROUPS = {
    'experiment': ('run_id', 'protocol_version', 'workload_version', 'hypothesis',
                   'primary_metric', 'rejection_rule', 'planned_sample_count',
                   'workload_manifest_sha256'),
    'software': ('revision', 'dirty_diff_sha256', 'binary_hashes', 'loaded_library_hashes',
                 'build_commands', 'compiler_flags', 'dependency_versions', 'toolchain_versions'),
    'artifacts': ('target', 'assistant', 'projector', 'tensor_manifest_sha256', 'derivative_lineage'),
    'request': ('rendered_sha256', 'tokens_sha256', 'occupied_input_tokens', 'template_sha256',
                'tokenizer_sha256', 'special_tokens', 'thinking', 'sampler_order',
                'sampler_parameters', 'seed', 'stop_rules', 'max_output_tokens'),
    'execution': ('capacity_tokens', 'occupied_after_prefill', 'occupied_after_decode',
                  'local_window', 'kv_formats', 'kv_placement', 'kv_allocation', 'checkpoints',
                  'context_shifting', 'batch', 'microbatch', 'threads', 'affinity',
                  'graph_cache', 'pipeline_cache', 'expert_placement', 'expert_cache_budget_bytes'),
    'mtp': ('enabled', 'assistant_revision', 'draft_limits', 'acceptance_policy',
            'target_verification_width', 'kv_sharing', 'rollback_settings'),
    'environment': ('hardware_class', 'kernel', 'driver_versions', 'runtime_versions',
                    'library_versions', 'selected_device', 'power_policy', 'telemetry_cadence_s',
                    'start_temperature_c', 'end_temperature_c', 'background_activity',
                    'resource_limits', 'run_window_review'),
    'measurement': ('timing_boundaries', 'warmup_state', 'cache_state', 'raw_samples_sha256',
                    'failure_reason', 'instrumentation_overhead', 'analysis_version'),
}
SHA_FIELDS = {'dirty_diff_sha256', 'workload_manifest_sha256', 'tensor_manifest_sha256',
              'rendered_sha256', 'tokens_sha256', 'template_sha256', 'tokenizer_sha256',
              'raw_samples_sha256'}
POSITIVE_FIELDS = {'planned_sample_count', 'occupied_input_tokens', 'max_output_tokens',
                   'capacity_tokens', 'local_window', 'batch', 'microbatch', 'threads',
                   'target_verification_width', 'telemetry_cadence_s'}
BOOL_FIELDS = {'thinking', 'enabled', 'context_shifting', 'kv_sharing'}
# These are structural prerequisites for a proposed performance comparison,
# not an automatic GO or approval to execute it.
REQUIRED_FOR_COMPARISON = {
    'experiment': GROUPS['experiment'],
    'software': ('revision', 'dirty_diff_sha256', 'binary_hashes', 'loaded_library_hashes',
                 'build_commands', 'compiler_flags', 'dependency_versions', 'toolchain_versions'),
    'artifacts': ('target', 'tensor_manifest_sha256'),
    'request': GROUPS['request'],
    'execution': ('capacity_tokens', 'local_window', 'kv_formats', 'kv_placement',
                  'batch', 'microbatch', 'threads', 'affinity', 'expert_placement'),
    'mtp': ('enabled',),
    'environment': ('hardware_class', 'kernel', 'driver_versions', 'runtime_versions',
                    'selected_device', 'power_policy', 'telemetry_cadence_s',
                    'resource_limits', 'run_window_review'),
    'measurement': ('timing_boundaries', 'warmup_state', 'cache_state', 'analysis_version'),
}


def sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_value(group, name, value):
    if value is None:
        raise ValueError('omit unknown fields instead of supplying unreasoned null')
    if name in SHA_FIELDS and not sha256(value):
        raise ValueError('invalid identity hash')
    if name == 'revision' and (not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value)):
        raise ValueError('full software revision required')
    if name in POSITIVE_FIELDS and (type(value) not in (int, float) or value <= 0):
        raise ValueError('positive identity value required')
    if name in POSITIVE_FIELDS - {'telemetry_cadence_s'} and type(value) is not int:
        raise ValueError('integer identity count required')
    if name in BOOL_FIELDS and type(value) is not bool:
        raise ValueError('boolean identity value required')
    if name == 'seed' and (type(value) is not int or value < 0):
        raise ValueError('nonnegative integer seed required')
    if name in ('sampler_order', 'stop_rules') and (not isinstance(value, list) or any(not isinstance(v, str) for v in value)):
        raise ValueError('string list required')
    if name == 'sampler_order' and not value:
        raise ValueError('sampler order must be explicit')
    if name == 'sampler_parameters' and not isinstance(value, dict):
        raise ValueError('sampler parameter map required')
    if name in ('binary_hashes', 'loaded_library_hashes'):
        if not isinstance(value, dict) or not value or any(not sha256(v) for v in value.values()):
            raise ValueError('explicit file hash map required')
    if group == 'artifacts' and name in ('target', 'assistant', 'projector'):
        if not isinstance(value, dict) or set(value) != {'sha256', 'size_bytes'} or not sha256(value['sha256']) or type(value['size_bytes']) is not int or value['size_bytes'] <= 0:
            raise ValueError('artifact hash and positive size required')
    if isinstance(value, str) and not value.strip():
        raise ValueError('empty identity string')
    canonical(value)  # Reject non-JSON and nonfinite values, including nested ones.


def make_identity(provided, *, missing_reason):
    """provided[group][field] supplies a value; omitted values become explicit null.

    Nonapplicable values can instead be omitted with a field-specific reason via
    annotate_missing. Empty supplied strings do not stand in for unknown facts.
    """
    if not isinstance(missing_reason, str) or not missing_reason.strip():
        raise ValueError('reason for missing identity fields required')
    if set(provided) - set(GROUPS):
        raise ValueError('unknown identity group')
    fields = {}
    for group, names in GROUPS.items():
        source = provided.get(group, {})
        if not isinstance(source, dict) or set(source) - set(names):
            raise ValueError('unknown identity field')
        fields[group] = {}
        for name in names:
            if name in source:
                validate_value(group, name, source[name])
                fields[group][name] = {'value': source[name], 'reason': None}
            else:
                fields[group][name] = {'value': None, 'reason': missing_reason}
    record = {'schema': SCHEMA, 'fields': fields}
    record['identity_sha256'] = digest(record)
    return record


def annotate_missing(record, reasons):
    """Return a new identity with precise missing/not-applicable explanations."""
    import json
    result = json.loads(canonical(record))
    validate_identity(result)
    for group, fields in reasons.items():
        for name, reason in fields.items():
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError('nonempty missing reason required')
            entry = result['fields'][group][name]
            if entry['value'] is not None:
                raise ValueError('cannot mark a known value missing')
            entry['reason'] = reason
    del result['identity_sha256']
    result['identity_sha256'] = digest(result)
    return result


def validate_identity(record):
    if set(record) != {'schema', 'fields', 'identity_sha256'} or record['schema'] != SCHEMA:
        raise ValueError('invalid identity schema')
    unhashed = {k: v for k, v in record.items() if k != 'identity_sha256'}
    if digest(unhashed) != record['identity_sha256'] or set(record['fields']) != set(GROUPS):
        raise ValueError('identity checksum/groups mismatch')
    for group, names in GROUPS.items():
        if set(record['fields'][group]) != set(names):
            raise ValueError('identity fields mismatch')
        for name, entry in record['fields'][group].items():
            if not isinstance(entry, dict) or set(entry) != {'value', 'reason'}:
                raise ValueError('invalid identity entry')
            if entry['value'] is None:
                if not isinstance(entry['reason'], str) or not entry['reason'].strip():
                    raise ValueError('missing identity reason')
            else:
                if entry['reason'] is not None:
                    raise ValueError('known identity value must not have missing reason')
                validate_value(group, name, entry['value'])
    return comparison_gaps(record)


def comparison_gaps(record):
    return [group + '.' + name for group, names in REQUIRED_FOR_COMPARISON.items()
            for name in names if record['fields'][group][name]['value'] is None]
