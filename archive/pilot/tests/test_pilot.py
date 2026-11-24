import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "twenty_questions_pilot.py"


class PilotTests(unittest.TestCase):
    def run_runner(self, *args):
        return subprocess.run(
            [sys.executable, str(RUNNER), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_mock_round_robin_creates_and_validates_48_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mock.jsonl"
            run = self.run_runner("--mock", "--repeats", "1", "--results-file", str(output))
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 48)
            games = [json.loads(line) for line in lines]
            self.assertEqual(len({game["game_id"] for game in games}), 48)
            self.assertEqual(len({(game["answerer"], game["guesser"]) for game in games}), 16)
            models = {
                game["answerer_model"] for game in games
            } | {
                game["guesser_model"] for game in games
            }
            self.assertEqual(models, {
                "Qwen/Qwen3-8B",
                "google/gemma-3-12b-it",
                "mistralai/Mistral-Nemo-Instruct-2407",
                "ibm-granite/granite-3.3-8b-instruct",
            })

    def test_validate_only_rejects_incomplete_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "incomplete.jsonl"
            output.write_text(
                json.dumps({"game_id": "one", "answerer": "A", "guesser": "B", "secret": "banana"})
                + "\n",
                encoding="utf-8",
            )
            run = self.run_runner("--validate-only", str(output))
            self.assertEqual(run.returncode, 1)
            self.assertIn("Validation failed", run.stdout)

    def test_rejects_results_from_another_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "old-cohort.jsonl"
            output.write_text(
                json.dumps({
                    "game_id": "old|cohort|record",
                    "answerer": "Qwen3 0.6B (HF local)",
                    "guesser": "Llama (Ollama)",
                }) + "\n",
                encoding="utf-8",
            )
            run = self.run_runner("--mock", "--results-file", str(output))
            self.assertEqual(run.returncode, 1)
            self.assertIn("contains another cohort", run.stdout)

    def test_resume_skips_existing_game_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mock.jsonl"
            first = self.run_runner("--mock", "--repeats", "1", "--results-file", str(output))
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)

            second = self.run_runner("--mock", "--repeats", "1", "--results-file", str(output))
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("playing 0 more", second.stdout)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 48)


if __name__ == "__main__":
    unittest.main()
