"""Commit research outputs as they change, so git history is the trail of the shared brain."""

import fcntl
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def commit(paths: list[Path], message: str) -> str | None:
    """Commit only `paths`; other edited or staged files are left alone. Returns the short
    hash, or None if nothing changed or git failed. A lock file in the git dir serializes
    commits from research runs going in parallel."""
    lock = _git("rev-parse", "--git-path", "shared-brain.lock")
    if lock.returncode:
        log.warning("git commit skipped: %s", lock.stderr.strip())
        return None
    with open(lock.stdout.strip(), "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        names = [str(p) for p in paths if p.exists()]
        if not names:
            return None
        if (add := _git("add", "--", *names)).returncode:
            log.warning("git add failed: %s", add.stderr.strip())
            return None
        if _git("diff", "--cached", "--quiet", "--", *names).returncode == 0:
            return None
        if (done := _git("commit", "-q", "-m", message, "--", *names)).returncode:
            log.warning("git commit failed: %s", (done.stderr or done.stdout).strip())
            return None
        return _git("rev-parse", "--short", "HEAD").stdout.strip()
