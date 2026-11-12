"""Parse the checksummed event stream emitted by the sampled-token observer."""
import re


READY = b'ROSEVINE_SAMPLER_HOOK_V1 READY'
ERROR_PREFIX = b'ROSEVINE_SAMPLER_HOOK_V1 ERROR '
TOKEN_PREFIX = b'ROSEVINE_SAMPLED_TOKEN_V1 '
DONE_PREFIX = b'ROSEVINE_SAMPLER_HOOK_V1 DONE '
INTEGER = re.compile(rb'(?:0|[1-9][0-9]*)\Z')


def _decimal(value):
    if not INTEGER.fullmatch(value):
        raise ValueError('malformed sampled-token event integer')
    return int(value)


def parse(stderr, *, vocabulary_size, eos_token_id, max_output_tokens):
    """Recover exact sampled IDs; unrelated native diagnostics are ignored.

    A missing hook marker, dropped/reordered token event, observer write error,
    early non-EOS termination or output count mismatch is an invalid receipt.
    """
    if not isinstance(stderr, bytes) or len(stderr) > 8 * 1024 * 1024:
        raise ValueError('sampled-token event stream exceeds its bound')
    if (type(vocabulary_size) is not int or vocabulary_size <= 1
            or type(eos_token_id) is not int or not 0 <= eos_token_id < vocabulary_size
            or type(max_output_tokens) is not int or max_output_tokens <= 0):
        raise ValueError('invalid sampled-token policy')
    tokens, ready_count, done, errors = [], 0, None, []
    for raw_line in stderr.splitlines():
        line = raw_line[:-1] if raw_line.endswith(b'\r') else raw_line
        if line == READY:
            if ready_count or tokens or done is not None:
                raise ValueError('sampler observer ready event is out of order')
            ready_count += 1
        elif line.startswith(ERROR_PREFIX):
            errors.append(line[len(ERROR_PREFIX):])
        elif line.startswith(TOKEN_PREFIX):
            if ready_count != 1 or done is not None:
                raise ValueError('sampled-token event is outside the observer lifetime')
            fields = line[len(TOKEN_PREFIX):].split(b' ')
            if len(fields) != 2:
                raise ValueError('malformed sampled-token event')
            index, token = map(_decimal, fields)
            if index != len(tokens) or token >= vocabulary_size or len(tokens) >= max_output_tokens:
                raise ValueError('sampled-token event order or value is invalid')
            tokens.append(token)
        elif line.startswith(DONE_PREFIX):
            if ready_count != 1 or done is not None:
                raise ValueError('sampled-token completion event is out of order or duplicated')
            fields = line[len(DONE_PREFIX):].split(b' ')
            if len(fields) != 2 or fields[1] not in (b'0', b'1'):
                raise ValueError('malformed sampled-token completion event')
            done = (_decimal(fields[0]), fields[1] == b'1')
        elif (line.startswith(b'ROSEVINE_SAMPLER_HOOK_V1')
              or line.startswith(b'ROSEVINE_SAMPLED_TOKEN_V1')):
            raise ValueError('malformed sampled-token observer event')
    if ready_count != 1 or done is None or errors or done[1] or done[0] != len(tokens) or not tokens:
        raise ValueError('sampled-token observer did not produce a complete receipt')
    eos_positions = [index for index, token in enumerate(tokens) if token == eos_token_id]
    if eos_positions:
        if eos_positions != [len(tokens) - 1]:
            raise ValueError('EOS must be the final sampled token')
        finish_reason = 'eos'
    elif len(tokens) == max_output_tokens:
        finish_reason = 'limit'
    else:
        raise ValueError('sampled output ended before EOS or its declared token limit')
    return {'token_ids': tokens, 'finish_reason': finish_reason}
