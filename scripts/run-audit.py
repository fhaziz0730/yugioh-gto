"""Run real-core ruling/privacy scenarios and save a machine-readable report."""
import json
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class AuditResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenarios = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.scenarios.append({"name": test.id(), "status": "passed"})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.scenarios.append({"name": test.id(), "status": "failed", "detail": self._exc_info_to_string(error, test)})

    def addError(self, test, error):
        super().addError(test, error)
        self.scenarios.append({"name": test.id(), "status": "error", "detail": self._exc_info_to_string(error, test)})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.scenarios.append({"name": test.id(), "status": "skipped", "detail": reason})


def main():
    os.environ["YGO_NATIVE_AUDIT"] = "1"
    root = Path(__file__).resolve().parent.parent
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_native_rules.py")
    result = unittest.TextTestRunner(verbosity=2, resultclass=AuditResult).run(suite)
    from yugioh_gto.training import digest
    engine = root / "vendor/ygo-agent"
    report = {"passed": result.wasSuccessful() and not result.skipped,
              "tests_run": result.testsRun, "scenarios": result.scenarios,
              "engine": json.loads((engine / "integration.json").read_text()),
              "native_sha256": digest(engine / "ygoenv/ygoenv/ygopro/ygopro_ygoenv.so"),
              "scope": "Controlled Elfnote rulings and visibility; not exhaustive TCG certification"}
    output = root / "runs/rules-audit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
