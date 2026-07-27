#!/usr/bin/env python3
"""
test_wip_audit.py — unit tests for the read-only WIP/backup exposure reporter.

Covers:
  - never-versioned (untracked) detection: the highest-exposure class
  - age bucketing from mtime, with backdated files (stale vs ageing vs fresh)
  - deleted paths: age falls back to the last commit that touched them
  - stranded rename tails: a deletion whose basename lives on elsewhere
  - commits on no remote, incl. a branch with NO upstream configured
  - `status: active` untracked handoff notes are surfaced
  - editor/plugin state (.obsidian/) is ignored by design
  - clean repo → empty, honest report
  - THE INVARIANT: the audit never mutates git state or the working tree

Run:  python3 tools/test_wip_audit.py       (stdlib unittest, no pytest needed)
      python3 -m pytest tools/test_wip_audit.py
No network. Every test builds a throwaway git repo in a temp dir.
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wip_audit  # noqa: E402

DAY = 86400.0


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )
    return out.stdout


def write(repo: Path, rel: str, text: str = "x", age_days: float = 0.0) -> Path:
    fp = repo / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(text, encoding="utf-8")
    if age_days:
        t = time.time() - age_days * DAY
        os.utime(fp, (t, t))
    return fp


def make_repo(tmp: str, name: str = "work") -> Path:
    """A repo with one commit on `main`, plus a bare 'remote' wired as origin."""
    repo = Path(tmp) / name
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@t.t")
    git(repo, "config", "user.name", "T")
    git(repo, "config", "commit.gpgsign", "false")
    write(repo, "seed.txt", "seed")
    git(repo, "add", "seed.txt")
    git(repo, "commit", "-qm", "seed")

    remote = Path(tmp) / f"{name}-remote.git"
    git(repo, "init", "-q", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "-u", "origin", "main")
    return repo


class TestNeverVersioned(unittest.TestCase):
    def test_untracked_is_never_versioned_and_aged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "newthing/gateway.py", "code", age_days=18)
            a = wip_audit.audit(repo)

            self.assertEqual(a["never_versioned_count"], 1)
            self.assertIn("newthing/gateway.py", a["untracked"])
            item = next(i for i in a["items"] if i["path"] == "newthing/gateway.py")
            self.assertTrue(item["never_versioned"])
            self.assertAlmostEqual(item["age_days"], 18, delta=0.5)

    def test_modified_tracked_file_is_not_never_versioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "seed.txt", "changed", age_days=3)
            a = wip_audit.audit(repo)

            self.assertEqual(a["never_versioned_count"], 0)
            item = next(i for i in a["items"] if i["path"] == "seed.txt")
            self.assertFalse(item["never_versioned"])
            self.assertEqual(len(a["items"]), 1)


class TestAgeBuckets(unittest.TestCase):
    def test_stale_ageing_and_fresh_are_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "a_stale.txt", age_days=20)   # >= 14 → stale
            write(repo, "b_ageing.txt", age_days=9)   # 7..13 → ageing
            write(repo, "c_fresh.txt", age_days=1)    # < 7  → neither
            a = wip_audit.audit(repo)

            self.assertEqual({i["path"] for i in a["stale"]}, {"a_stale.txt"})
            self.assertEqual({i["path"] for i in a["warn"]}, {"b_ageing.txt"})
            self.assertAlmostEqual(a["oldest_age_days"], 20, delta=0.5)

    def test_stale_threshold_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "wip.txt", age_days=20)
            self.assertEqual(len(wip_audit.audit(repo, stale_days=30)["stale"]), 0)
            self.assertEqual(len(wip_audit.audit(repo, stale_days=14)["stale"]), 1)

    def test_warn_band_excludes_stale_and_fresh(self):
        self.assertTrue(wip_audit.warn_band(7.0, 14))
        self.assertTrue(wip_audit.warn_band(13.9, 14))
        self.assertFalse(wip_audit.warn_band(14.0, 14))
        self.assertFalse(wip_audit.warn_band(6.9, 14))


class TestDeletionsAndRenameTails(unittest.TestCase):
    def test_deleted_path_age_falls_back_to_last_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "data/old.csv", "1,2")
            git(repo, "add", "data/old.csv")
            git(repo, "commit", "-qm", "add csv")
            (repo / "data/old.csv").unlink()

            a = wip_audit.audit(repo)
            item = next(i for i in a["items"] if i["path"] == "data/old.csv")
            # No mtime available; age comes from the commit and must be a real number.
            self.assertIsNotNone(item["age_days"])
            self.assertGreaterEqual(item["age_days"], 0.0)

    def test_stranded_rename_tail_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "old/tape.csv", "1")
            write(repo, "new/tape.csv", "1")
            git(repo, "add", "old/tape.csv", "new/tape.csv")
            git(repo, "commit", "-qm", "both paths")
            (repo / "old/tape.csv").unlink()   # deletion left unstaged

            a = wip_audit.audit(repo)
            self.assertEqual(len(a["rename_tails"]), 1)
            tail = a["rename_tails"][0]
            self.assertEqual(tail["path"], "old/tape.csv")
            self.assertEqual(tail["twins"][0]["path"], "new/tape.csv")
            self.assertTrue(tail["twins"][0]["tracked"])
            self.assertFalse(tail["dest_unversioned"])

    def test_twin_in_a_gitignored_tree_is_found_and_flagged_unversioned(self):
        """The motivating real case: a reorg moved data into an ignored tree, so
        `git ls-files` cannot see the destination and the bytes left git."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, ".gitignore", "newdata/\n")
            write(repo, "olddata/tape.csv", "1")
            git(repo, "add", ".gitignore", "olddata/tape.csv")
            git(repo, "commit", "-qm", "tracked at old path")
            write(repo, "newdata/tape.csv", "1")     # ignored destination
            (repo / "olddata/tape.csv").unlink()     # deletion left unstaged

            a = wip_audit.audit(repo)
            self.assertEqual(len(a["rename_tails"]), 1)
            tail = a["rename_tails"][0]
            self.assertEqual(tail["twins"][0]["path"], "newdata/tape.csv")
            self.assertFalse(tail["twins"][0]["tracked"])
            self.assertTrue(tail["twins"][0]["ignored"])
            self.assertTrue(tail["dest_unversioned"])
            self.assertIn("unversioned (ignored)", wip_audit.render(a))

    def test_deletion_with_no_twin_is_not_a_rename_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            (repo / "seed.txt").unlink()
            a = wip_audit.audit(repo)
            self.assertEqual(a["rename_tails"], [])


class TestBackupExposure(unittest.TestCase):
    def test_unpushed_commits_counted_against_upstream_and_remotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            for n in range(3):
                write(repo, f"f{n}.txt", str(n))
                git(repo, "add", f"f{n}.txt")
                git(repo, "commit", "-qm", f"c{n}")

            a = wip_audit.audit(repo)
            main = next(b for b in a["branches"] if b["branch"] == "main")
            self.assertEqual(main["ahead_of_upstream"], 3)
            self.assertEqual(main["commits_on_no_remote"], 3)
            self.assertEqual(a["commits_on_no_remote"], 3)

    def test_branch_with_no_upstream_still_reports_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            git(repo, "checkout", "-q", "-b", "solo")
            write(repo, "solo.txt", "s")
            git(repo, "add", "solo.txt")
            git(repo, "commit", "-qm", "solo work")

            a = wip_audit.audit(repo)
            solo = next(b for b in a["branches"] if b["branch"] == "solo")
            self.assertEqual(solo["upstream"], "")
            # No upstream → "ahead" is unknowable, but exposure is not.
            self.assertIsNone(solo["ahead_of_upstream"])
            self.assertEqual(solo["commits_on_no_remote"], 1)

    def test_fully_pushed_repo_reports_zero_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            a = wip_audit.audit(repo)
            self.assertEqual(a["commits_on_no_remote"], 0)
            self.assertEqual(a["items"], [])
            self.assertIsNone(a["oldest_age_days"])
            self.assertIn("_Clean working tree._", wip_audit.render(a))


class TestActiveNotesAndIgnores(unittest.TestCase):
    def test_untracked_active_handoff_is_surfaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "brain/handoffs/live.md", "---\nstatus: active\n---\nbody")
            write(repo, "brain/handoffs/done.md", "---\nstatus: closed\n---\nbody")
            a = wip_audit.audit(repo)
            self.assertEqual(a["active_untracked_notes"], ["brain/handoffs/live.md"])

    def test_tracked_active_handoff_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "brain/handoffs/live.md", "---\nstatus: active\n---\nbody")
            git(repo, "add", "brain/handoffs/live.md")
            git(repo, "commit", "-qm", "handoff")
            a = wip_audit.audit(repo)
            self.assertEqual(a["active_untracked_notes"], [])

    def test_obsidian_plugin_state_is_ignored_by_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, ".obsidian/plugins/x/data.json", "{}", age_days=30)
            a = wip_audit.audit(repo)
            self.assertEqual(a["items"], [])
            self.assertEqual(a["ignored_count"], 1)

    def test_threads_group_by_first_two_path_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "polymarket/execution/a.py")
            write(repo, "polymarket/execution/b.py")
            write(repo, "polymarket/research/c.py")
            a = wip_audit.audit(repo)
            self.assertEqual(len(a["by_thread"]["polymarket/execution"]), 2)
            self.assertEqual(len(a["by_thread"]["polymarket/research"]), 1)

    def test_depth_two_file_groups_under_its_directory_not_itself(self):
        """`meetings/README.md` is not its own thread — that is noise."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "meetings/README.md")
            write(repo, "meetings/notes.md")
            a = wip_audit.audit(repo)
            self.assertEqual(len(a["by_thread"]["meetings"]), 2)
            self.assertNotIn("meetings/README.md", a["by_thread"])

    def test_root_level_file_gets_root_label(self):
        self.assertEqual(wip_audit.thread_of("TODO.md"), "(root)")
        self.assertEqual(wip_audit.thread_of("meetings/README.md"), "meetings")
        self.assertEqual(wip_audit.thread_of("a/b/c/d.py"), "a/b")


class TestReadOnlyInvariant(unittest.TestCase):
    def test_audit_and_render_never_mutate_git_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "dirty.txt", "d", age_days=20)
            write(repo, "brain/handoffs/live.md", "---\nstatus: active\n---\n")
            (repo / "seed.txt").unlink()

            before_status = git(repo, "status", "--porcelain")
            before_head = git(repo, "rev-parse", "HEAD")
            before_reflog = git(repo, "reflog", "--format=%H %gs")
            before_stash = git(repo, "stash", "list")

            a = wip_audit.audit(repo)
            wip_audit.render(a)

            self.assertEqual(git(repo, "status", "--porcelain"), before_status)
            self.assertEqual(git(repo, "rev-parse", "HEAD"), before_head)
            self.assertEqual(git(repo, "reflog", "--format=%H %gs"), before_reflog)
            self.assertEqual(git(repo, "stash", "list"), before_stash)

    def test_dry_run_writes_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "dirty.txt", "d")
            rc = wip_audit.main(["--repo", str(repo), "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertFalse((repo / wip_audit.REPORT_REL).exists())

    def test_main_writes_report_and_is_idempotent_on_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "dirty.txt", "d", age_days=20)
            before_status = git(repo, "status", "--porcelain", "-uall")

            rc = wip_audit.main(["--repo", str(repo)])
            self.assertEqual(rc, 0)
            report = repo / wip_audit.REPORT_REL
            self.assertTrue(report.exists())
            text = report.read_text(encoding="utf-8")
            self.assertIn("# WIP Exposure Audit", text)
            self.assertIn("dirty.txt", text)
            self.assertIn("Finds exposure; does not fix it", text)
            # The report itself lands in brain/generated/ (git-ignored in the real
            # repo), so the only permitted status delta is that one report path.
            # -uall stops git collapsing the new dir to a bare `?? brain/`.
            after = git(repo, "status", "--porcelain", "-uall")
            delta = set(after.splitlines()) - set(before_status.splitlines())
            self.assertEqual(delta, {f"?? {wip_audit.REPORT_REL}"})

    def test_non_repo_path_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(wip_audit.main(["--repo", tmp]), 2)


class TestHeadline(unittest.TestCase):
    def test_headline_reports_counts_and_oldest_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            write(repo, "u.txt", "u", age_days=18)
            write(repo, "seed.txt", "changed", age_days=2)
            h = wip_audit.headline(wip_audit.audit(repo))
            self.assertIn("2 uncommitted paths", h)
            self.assertIn("1 never versioned", h)
            self.assertIn("oldest 18d", h)

    def test_headline_handles_clean_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(tmp)
            h = wip_audit.headline(wip_audit.audit(repo))
            self.assertIn("0 uncommitted paths", h)
            self.assertIn("oldest n/a", h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
