"""Synthetic layout and corruption tests, independent of model availability."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('gguf_audit', Path(__file__).resolve().parents[2] / 'scripts/gguf_audit.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def string(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


def fixture(kind=2, width=32, offset=0, payload=18):
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)
    header += string('weight') + struct.pack('<IQIQ', 1, width, kind, offset)
    return header + bytes((-len(header)) % 32) + bytes(payload)


class AuditTests(unittest.TestCase):
    def check_data(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.gguf'
            path.write_bytes(data)
            return AUDIT.audit(path, full_hash=True)

    def test_block_layout_and_hash(self):
        import hashlib
        data = fixture()
        result = self.check_data(data)
        self.assertEqual(result['tensors'][0]['bytes'], 18)
        self.assertEqual(result['sha256'], hashlib.sha256(data).hexdigest())

    def test_f32_layout(self):
        self.assertEqual(self.check_data(fixture(0, 3, payload=12))['tensors'][0]['bytes'], 12)

    def test_reject_corruptions(self):
        for data in [fixture()[:-1], fixture(width=31), fixture(offset=1), fixture(kind=1), b'GGUF', fixture(width=0)]:
            with self.subTest(length=len(data)):
                with self.assertRaises((AUDIT.InvalidGGUF, UnicodeError)):
                    self.check_data(data)

    def test_overlapping_ranges(self):
        tensor = string('a') + struct.pack('<IQIQ', 1, 32, 2, 0)
        other = string('b') + struct.pack('<IQIQ', 1, 32, 2, 0)
        header = b'GGUF' + struct.pack('<IQQ', 3, 2, 0) + tensor + other
        with self.assertRaises(AUDIT.InvalidGGUF):
            self.check_data(header + bytes((-len(header)) % 32) + bytes(18))


if __name__ == '__main__':
    unittest.main()
