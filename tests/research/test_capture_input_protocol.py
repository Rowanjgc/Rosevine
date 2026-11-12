"""Compile the llama-independent native capture framing decoder and exercise boundaries."""
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CaptureInputProtocolTests(unittest.TestCase):
    def test_native_decoder_accepts_v3_long_selected_context_and_rejects_corruption(self):
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('existing g++ compiler unavailable; no installation attempted')
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / 'capture-input-protocol-test'
            source = ROOT / 'tests/research/capture_input_protocol_test.cpp'
            result = subprocess.run(
                [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                 '-pedantic', '-I' + str(ROOT / 'research'), str(source), '-o', str(binary)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace'))
            run = subprocess.run([str(binary)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=5, check=False)
            self.assertEqual(run.returncode, 0, run.stderr.decode('utf-8', 'replace'))


if __name__ == '__main__':
    unittest.main()
