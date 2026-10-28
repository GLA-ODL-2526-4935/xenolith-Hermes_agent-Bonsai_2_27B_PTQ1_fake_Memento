"""Small synthetic GGUF tests; no real weights, tensor libraries or inference."""
from pathlib import Path
import struct
import tempfile
import unittest
import mtp_artifact as tool


def string(text):
    data = text.encode()
    return struct.pack('<Q', len(data)) + data


def fixture(path, label, head=b'H' * 32, backbone=b'B' * 32):
    header = b'GGUF' + struct.pack('<IQQ', 3, 2, 1)
    header += string('general.name') + struct.pack('<I', 8) + string(label)
    for name, offset in [('token_embd.weight', 0), ('blk.64.nextn.hnorm.weight', 32)]:
        header += string(name) + struct.pack('<IQIQ', 1, 8, 0, offset)
    header += bytes((-len(header)) % 32)
    path.write_bytes(header + backbone + head)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base, self.trained, self.package, self.out = [self.root / name for name in ('base.gguf', 'trained.gguf', 'head.zip', 'out.gguf')]
        fixture(self.base, 'base')
        fixture(self.trained, 'trained with a deliberately different header length' * 2, head=b'T' * 32)

    def test_exact_roundtrip_with_header_shift(self):
        tool.pack(self.base, self.trained, self.package)
        tool.apply(self.base, self.package, self.out)
        self.assertEqual(self.trained.read_bytes(), self.out.read_bytes())
        self.assertEqual(self.base.read_bytes()[-32:], b'H' * 32)

    def test_backbone_change_refused(self):
        fixture(self.trained, 'modified', backbone=b'X' * 32)
        with self.assertRaisesRegex(ValueError, 'Backbone changed'):
            tool.pack(self.base, self.trained, self.package)
        self.assertFalse(self.package.exists())
        self.assertFalse(self.package.with_name('head.zip.partial').exists())

    def test_wrong_base_refused(self):
        tool.pack(self.base, self.trained, self.package)
        fixture(self.base, 'wrong')
        with self.assertRaisesRegex(ValueError, 'base SHA256 mismatch'):
            tool.apply(self.base, self.package, self.out)
        self.assertFalse(self.out.exists())

    def test_existing_output_preserved(self):
        self.out.write_bytes(b'keep')
        with self.assertRaises(ValueError):
            tool.apply(self.base, self.package, self.out)
        self.assertEqual(self.out.read_bytes(), b'keep')

    def test_existing_partial_preserved(self):
        partial = self.out.with_name('out.gguf.partial')
        partial.write_bytes(b'keep')
        with self.assertRaises(ValueError):
            tool.apply(self.base, self.package, self.out)
        self.assertEqual(partial.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
