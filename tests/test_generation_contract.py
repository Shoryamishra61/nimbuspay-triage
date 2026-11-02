"""Verification of strict generation contract and module boundary rules."""

import inspect
import src.triage.generation as gen_module


def test_generation_module_boundaries():
    """Ensure generation.py has no dependency on evidence.py, clean.py, or audit.py."""
    source = inspect.getsource(gen_module)
    assert "import evidence" not in source and "from .evidence" not in source and "from src.triage.evidence" not in source, "generation.py must NEVER import evidence.py"
    assert "import clean" not in source and "from .clean" not in source and "from src.triage.clean" not in source, "generation.py must NEVER import clean.py"
    assert "import audit" not in source and "from .audit" not in source and "from src.triage.audit" not in source, "generation.py must NEVER import audit.py"
    assert "re.sub(" not in source, "generation.py must NOT do regex replacements"
    assert "json.loads(" not in source, "generation.py must NOT parse or rewrite JSON outputs"


def test_generation_postprocessing_restriction():
    """Ensure raw string output is returned with only .strip()."""
    source = inspect.getsource(gen_module.generate_single)
    assert ".strip()" in source
