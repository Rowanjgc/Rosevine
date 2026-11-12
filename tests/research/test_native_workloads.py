"""Native adapter failure retention without native tools or model files."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('native_workloads', ROOT / 'scripts/native_workloads.py')
N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(N)


class AdapterTests(unittest.TestCase):
    def test_failed_renderer_retains_private_input_and_log(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = N.NativeTextAdapter('renderer', 'tokenizer', 'model', 'template',
                                          {'tokenizer.ggml.tokens': ['BOS', 'EOS'],
                                           'tokenizer.ggml.bos_token_id': 0,
                                           'tokenizer.ggml.eos_token_id': 1}, Path(temp))
            with patch.object(N.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)):
                with self.assertRaises(ValueError):
                    adapter('synthetic fixture', {'thinking': True})
            retained = list(Path(temp).iterdir())
            self.assertEqual(len(retained), 1)
            self.assertTrue((retained[0] / 'input.json').is_file())
            self.assertTrue((retained[0] / 'native-stderr.txt').is_file())
            for file in retained[0].iterdir():
                self.assertEqual(file.stat().st_mode & 0o777, 0o600)
