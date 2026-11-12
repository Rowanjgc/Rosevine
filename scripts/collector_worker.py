#!/usr/bin/env python3
"""Bounded local JSON-pipe collector, independent of the supervised workload.

Only explicit trusted collector commands are launched. Responses are data, never
code/pickle. No device discovery, endpoint, model launch or workstation probe is
performed by this module. Kernel-blocked tasks may outlive SIGKILL; cleanup state
is retained and never represented as confirmed completion.
"""
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import selectors
import stat
import subprocess
import sys
import time

from benchmark_core import canonical, open_private, terminate_group, write_private_json


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate protocol key')
        result[key] = value
    return result


class ProcessCollector:
    def __init__(self, command, directory, *, env=None, timeout_s=.25,
                 max_response_bytes=65536, max_stderr_bytes=65536, grace_s=.05,
                 cleanup_timeout_s=1, artifact_prefix=''):
        if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
            raise ValueError('explicit collector command list required')
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('positive finite collector response deadline required')
        if any(type(v) is not int or v <= 0 for v in (max_response_bytes, max_stderr_bytes)):
            raise ValueError('positive collector byte limits required')
        if (not isinstance(artifact_prefix, str)
                or (artifact_prefix and (not artifact_prefix.isascii()
                                         or not artifact_prefix.replace('-', '').isalnum()
                                         or not artifact_prefix.endswith('-')))):
            raise ValueError('collector artifact prefix must be an empty or safe hyphenated name')
        for value, positive in ((grace_s, False), (cleanup_timeout_s, True)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError('finite collector cleanup deadlines required')
        self.command, self.directory, self.env = command, Path(directory), env
        self.artifact_prefix = artifact_prefix
        if self.directory.is_symlink() or not self.directory.is_dir() or stat.S_IMODE(self.directory.stat().st_mode) != 0o700:
            raise ValueError('private collector directory must have mode 0700')
        self.timeout_s, self.grace_s, self.cleanup_timeout_s = timeout_s, grace_s, cleanup_timeout_s
        self.max_response_bytes, self.max_stderr_bytes = max_response_bytes, max_stderr_bytes
        self.process, self.selector, self.errors = None, None, None
        self.stack = ExitStack()
        self.closed, self.failure, self.summary = False, None, None
        self.request_id, self.stderr_bytes, self.last_error_read = 0, 0, 0
        self.last_transport = None

    def __enter__(self):
        try:
            self.errors = self.stack.enter_context(open_private(
                self.directory / (self.artifact_prefix + 'collector-stderr.txt')))
            self.selector = self.stack.enter_context(selectors.DefaultSelector())
            self.process = subprocess.Popen(self.command, cwd=self.directory, env=self.env,
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, start_new_session=True,
                                            bufsize=0, umask=0o077)
            for name in ('stdin', 'stdout', 'stderr'):
                stream = getattr(self.process, name)
                self.stack.callback(stream.close)
                os.set_blocking(stream.fileno(), False)
            self.selector.register(self.process.stdout, selectors.EVENT_READ, 'response')
            self.selector.register(self.process.stderr, selectors.EVENT_READ, 'stderr')
            return self
        except BaseException as error:
            self.failure = 'collector startup failed: ' + type(error).__name__
            self.close()
            raise

    def fail(self, reason, prefix=b''):
        if self.failure is None:
            self.failure = reason
            if prefix:
                with open_private(self.directory / (self.artifact_prefix + 'collector-invalid-response.txt')) as stream:
                    stream.write(prefix[:self.max_response_bytes])
        return {'rejection': self.failure, 'collector_status': 'failed', 'request_id': self.request_id}

    def drain_errors(self):
        self.last_error_read = 0
        try:
            raw = os.read(self.process.stderr.fileno(), 65536)
        except BlockingIOError:
            return None
        if not raw:
            try:
                self.selector.unregister(self.process.stderr)
            except KeyError:
                pass
            return None
        self.last_error_read = len(raw)
        remaining = self.max_stderr_bytes - self.stderr_bytes
        self.errors.write(raw[:remaining])
        self.errors.flush()
        self.stderr_bytes += min(len(raw), remaining)
        return 'collector stderr limit reached' if len(raw) > remaining else None

    def sample(self, pid, elapsed_s, captured_bytes, *, timeout_s=None):
        if self.closed or self.process is None:
            raise ValueError('collector must be open')
        if self.failure is not None:
            return self.fail(self.failure)
        budget = self.timeout_s if timeout_s is None else min(self.timeout_s, timeout_s)
        if not math.isfinite(budget) or budget <= 0:
            return self.fail('collector response deadline exhausted')
        self.request_id += 1
        self.last_transport = None
        sent_ns = time.monotonic_ns()
        request = canonical({'request_id': self.request_id, 'pid': pid,
                             'elapsed_s': elapsed_s, 'captured_bytes': captured_bytes}) + b'\n'
        if len(request) > 4096:
            return self.fail('collector request limit reached')
        try:
            written = os.write(self.process.stdin.fileno(), request)
        except (BrokenPipeError, BlockingIOError):
            return self.fail('collector request unavailable')
        if written != len(request):
            return self.fail('collector request incomplete')
        deadline = sent_ns + int(budget * 1e9)
        response = bytearray()
        while True:
            remaining = (deadline - time.monotonic_ns()) / 1e9
            if remaining <= 0:
                return self.fail('collector response timeout', bytes(response))
            for key, _ in self.selector.select(remaining):
                if key.data == 'stderr':
                    reason = self.drain_errors()
                    if reason:
                        return self.fail(reason)
                    continue
                try:
                    raw = os.read(key.fileobj.fileno(), min(65536, self.max_response_bytes + 1))
                except BlockingIOError:
                    continue
                if not raw:
                    return self.fail('collector exited before response', bytes(response))
                response.extend(raw)
                if len(response) > self.max_response_bytes:
                    return self.fail('collector response limit reached', bytes(response))
                if b'\n' not in response:
                    continue
                if not response.endswith(b'\n') or response.count(b'\n') != 1:
                    return self.fail('collector response framing invalid', bytes(response))
                return self.decode_response(bytes(response), sent_ns)

    def decode_response(self, raw, sent_ns):
        try:
            response = json.loads(raw, object_pairs_hook=unique_object)
            if not isinstance(response, dict) or set(response) != {'request_id', 'started_ns', 'completed_ns', 'snapshot'}:
                return self.fail('collector response shape invalid', raw)
            if type(response['request_id']) is not int or response['request_id'] != self.request_id:
                return self.fail('collector response request mismatch', raw)
            before, after = response['started_ns'], response['completed_ns']
            if type(before) is not int or type(after) is not int or not sent_ns <= before <= after <= time.monotonic_ns():
                return self.fail('collector response timestamps invalid', raw)
            snapshot = response['snapshot']
            if not isinstance(snapshot, dict) or (snapshot.get('rejection') is not None and
                    (not isinstance(snapshot['rejection'], str) or not snapshot['rejection'].strip())):
                return self.fail('collector snapshot invalid', raw)
            canonical(snapshot)
            self.last_transport = {'request_id': self.request_id, 'requested_ns': sent_ns,
                                   'acquisition_started_ns': before, 'acquisition_completed_ns': after,
                                   'received_ns': time.monotonic_ns()}
            return snapshot
        except (ValueError, TypeError, UnicodeError, RecursionError):
            return self.fail('collector response JSON invalid', raw)

    def close(self):
        if self.closed:
            return self.summary
        self.closed = True
        if self.process is not None and self.process.poll() is not None:
            self.failure = self.failure or 'collector exited unexpectedly'
        cleanup = terminate_group(self.process, self.grace_s, self.cleanup_timeout_s) if self.process else None
        # Drain at most the remaining diagnostic cap plus one read to detect
        # truncation. Descriptors stay nonblocking even if cleanup is pending.
        if self.errors is not None and self.process is not None:
            try:
                for _ in range(self.max_stderr_bytes // 65536 + 2):
                    reason = self.drain_errors()
                    self.failure = self.failure or reason
                    if reason or not self.last_error_read:
                        break
            except (OSError, ValueError):
                self.failure = self.failure or 'collector stderr drain failed'
        self.stack.close()
        self.summary = {'status': 'incomplete' if self.failure or (cleanup and cleanup['pending']) else 'closed',
                        'failure': self.failure, 'requests': self.request_id,
                        'stderr_bytes': self.stderr_bytes, 'cleanup': cleanup}
        write_private_json(self.directory / (self.artifact_prefix + 'collector-result.json'), self.summary)
        return self.summary

    def __exit__(self, *args):
        self.close()


def serve(callback, *, max_request_bytes=4096):
    """Serve an explicit trusted callback in a separate process, never in parent.

    callback(pid_or_none, elapsed_s, captured_bytes) returns a snapshot dictionary.
    Monotonic acquisition times and request identity bind responses to requests.
    No stdout logging is allowed; diagnostics belong on stderr (bounded by parent).
    """
    previous_request = 0
    while True:
        raw = sys.stdin.buffer.readline(max_request_bytes + 1)
        if not raw:
            return
        if len(raw) > max_request_bytes or not raw.endswith(b'\n'):
            raise ValueError('collector request framing invalid')
        request = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(request, dict) or set(request) != {'request_id', 'pid', 'elapsed_s', 'captured_bytes'}:
            raise ValueError('collector request schema invalid')
        if type(request['request_id']) is not int or request['request_id'] <= 0 or (
                request['pid'] is not None and (type(request['pid']) is not int or request['pid'] <= 0)):
            raise ValueError('collector request identity invalid')
        if type(request['elapsed_s']) not in (int, float) or not math.isfinite(request['elapsed_s']) or request['elapsed_s'] < 0 or type(request['captured_bytes']) is not int or request['captured_bytes'] < 0:
            raise ValueError('collector request counters invalid')
        if request['request_id'] != previous_request + 1:
            raise ValueError('collector request order invalid')
        previous_request = request['request_id']
        before = time.monotonic_ns()
        try:
            snapshot = callback(request['pid'], request['elapsed_s'], request['captured_bytes'])
            if not isinstance(snapshot, dict):
                raise ValueError('collector snapshot must be an object')
            canonical(snapshot)
        except Exception as error:
            snapshot = {'rejection': 'collector callback failed', 'failure_category': type(error).__name__}
        response = {'request_id': request['request_id'], 'started_ns': before,
                    'completed_ns': time.monotonic_ns(), 'snapshot': snapshot}
        sys.stdout.buffer.write(canonical(response) + b'\n')
        sys.stdout.buffer.flush()
