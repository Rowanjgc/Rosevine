#!/usr/bin/env python3
"""Install a checksum-pinned scanner inside this repository, never globally."""
import hashlib
import io
import pathlib
import platform
import tarfile
import urllib.request

VERSION = '8.30.1'
# Official release checksum, independently fixed in source at setup time.
SHA256 = '551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb'


def main():
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise SystemExit('Pinned installer supports Linux x86_64; review and add a release checksum for this platform.')
    root = pathlib.Path(__file__).resolve().parents[1]
    dest = root / '.tools' / 'gitleaks'
    if dest.exists() and (root / '.tools' / 'gitleaks.sha256').exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == (root / '.tools' / 'gitleaks.sha256').read_text().strip():
        print('Scanner already installed; bootstrap will verify its version.')
        return
    url = f'https://github.com/gitleaks/gitleaks/releases/download/v{VERSION}/gitleaks_{VERSION}_linux_x64.tar.gz'
    data = urllib.request.urlopen(url, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit('Scanner checksum mismatch. Stop; review upstream release, do not bypass verification.')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        binary = archive.extractfile('gitleaks').read()
    dest.parent.mkdir(mode=0o700, exist_ok=True)
    dest.write_bytes(binary)
    dest.chmod(0o700)
    (dest.parent / 'gitleaks.sha256').write_text(hashlib.sha256(binary).hexdigest() + '\n')
    print('Installed checksum-verified Gitleaks ' + VERSION + ' locally.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, tarfile.TarError):
        raise SystemExit('Scanner installation failed. Check network, local permissions and pinned release availability; no hooks were bypassed.') from None
