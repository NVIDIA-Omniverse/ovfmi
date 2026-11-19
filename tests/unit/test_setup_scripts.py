# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def test_windows_embedded_python_survives_powershell_51_quoting():
    setup = (_ROOT / "demoapp" / "setup.ps1").read_text(encoding="utf-8")
    snippets = re.findall(r"(?s)\$code = @'\n(.*?)\n'@", setup)
    assert snippets
    for code in snippets:
        assert '"' not in code, (
            "Windows PowerShell 5.1 strips embedded double quotes when passing "
            "the here-string to Python's native -c argument"
        )
        compile(code, "setup embedded Python", "exec")


def test_setup_installs_pinned_packages_in_isolated_venv():
    ps = (_ROOT / "demoapp/setup.ps1").read_text()
    sh = (_ROOT / "demoapp/setup.sh").read_text()
    for script in (ps, sh):
        assert "requirements.txt" in script
        assert "requirements-experimental.txt" not in script
        assert "runtime_dependencies.py" in script
        assert "venv --system-site-packages" not in script
    assert 'Invoke-AppPip install -r $Requirements' in ps
    assert '"$APP_PYTHON" -m pip install -r "$REQUIREMENTS"' in sh
