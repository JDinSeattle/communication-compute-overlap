import itertools
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import paired_comparison
from run import pair, ready, validate_samples
import test_protocol as fixtures


class StatisticalTests(unittest.TestCase):
    def test_three_repeats_cannot_establish_benefit(self):
        self.assertEqual(paired_comparison([2., 2., 2.])["region"], "insufficient_repeats")

    def test_noisy_positive_median_remains_inconclusive(self):
        result = paired_comparison([.7, .9, 1.1, 1.2, 1.3, 1.5, 2.])
        self.assertEqual(result["median_paired_speedup"], 1.2)
        self.assertEqual(result["region"], "inconclusive")

    def test_separated_and_equivalent_regions(self):
        for xs, expected in [([1.2]*7, "beneficial"), ([.8]*7, "regressed"), ([1.]*7, "within_margin")]:
            self.assertEqual(paired_comparison(xs)["region"], expected)

    def test_interval_coverage_by_exhaustive_sign_patterns(self):
        # Independent oracle: all 2**n equiprobable signs around a continuous median.
        for n in range(6, 13):
            interval = paired_comparison(list(range(1, n+1)))["median_interval"]
            k = interval["order_index"]
            covered = sum(k <= sum(bits) <= n-k for bits in itertools.product((0, 1), repeat=n))
            self.assertAlmostEqual(covered / 2**n, interval["minimum_coverage"])
            self.assertGreaterEqual(covered / 2**n, .95)

    def test_bad_ratios_and_exact_threshold(self):
        for xs in ([], [True], [0], [math.inf], [math.nan]):
            with self.assertRaises(ValueError): paired_comparison(xs)
        self.assertEqual(paired_comparison([1.05]*7)["region"], "within_margin")


class RankAcceptanceTests(unittest.TestCase):
    def validate(self, logs):
        return validate_samples(logs, 1, 0, "overlap", 4096, 16384, 32)

    def test_rank_cardinality(self):
        for logs in ([], fixtures.ProtocolTests.logs()[:1], fixtures.ProtocolTests.logs()*2):
            with self.assertRaises(ValueError): self.validate(logs)

    def test_boolean_aliases_missing_checksum_and_reordered_result(self):
        for key, value in [("rank", False), ("correct", 1), ("sequence", True), ("checksum", None), ("wall_us", True)]:
            logs = fixtures.ProtocolTests.logs(); rows = [json.loads(s) for s in logs[0].splitlines()]
            rows[0][key] = value; logs[0] = "\n".join(map(json.dumps, rows))
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(logs)
        logs = fixtures.ProtocolTests.logs(); logs[0] = "\n".join(reversed(logs[0].splitlines()))
        with self.assertRaises(ValueError): self.validate(logs)

    def test_hardware_readiness_requires_both_peer_directions(self):
        payload = {"status":"ready", "device_count":2, "p2p_0_to_1":1, "p2p_1_to_0":1}
        good = {"returncode":0, "stdout":json.dumps(payload)}
        self.assertTrue(ready(good))
        self.assertEqual(self.validate([good["stdout"]+"\n"+s for s in fixtures.ProtocolTests.logs()]), [14])
        for p in ({**payload,"device_count":1}, {**payload,"p2p_1_to_0":0}, {**payload,"p2p_0_to_1":True}):
            self.assertFalse(ready({"returncode":0,"stdout":json.dumps(p)}))
        self.assertFalse(ready({"returncode":0,"stdout":""}))


class ProcessAcceptanceTests(unittest.TestCase):
    def run_pair(self, body, timeout=2):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); binary = root / "worker"
            binary.write_text("#!" + sys.executable + "\n" + body)
            binary.chmod(0o755)
            start = time.monotonic()
            result = pair(binary, root/"run", 4096, 16384, 32, "overlap", 1, 0, timeout)
            self.assertEqual(result, json.loads((root/"run/result.json").read_text()))
            self.assertLess(time.monotonic()-start, 5)
            return result

    def test_real_peer_exit_reaps_waiting_peer(self):
        result = self.run_pair("import sys,time\nif sys.argv[1]=='0': sys.exit(42)\ntime.sleep(30)\n")
        self.assertEqual(result["status"], "runtime_failure")
        self.assertEqual(result["returncodes"][0], 42)
        self.assertIsNotNone(result["returncodes"][1])
        self.assertIsNone(result["median_us"])

    def test_real_timeout_has_no_performance(self):
        result = self.run_pair("import time\ntime.sleep(30)\n", timeout=.1)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["critical_path_us"], [])

    def test_exit_zero_without_rank_evidence_fails(self):
        self.assertEqual(self.run_pair("pass\n")["status"], "invalid_evidence")

    def test_launch_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pair(Path(tmp)/"missing", Path(tmp)/"run", 4096, 16384, 32, "overlap", 1, 0, 1)
            self.assertEqual(result["status"], "launch_failure")
            self.assertIsNone(result["median_us"])

    def test_cli_analysis_with_synthetic_rank_workers(self):
        # Exercise real orchestration/serialization with CPU fixtures. This does
        # not assert that two GPUs exist or validate a PortChannel transfer.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); binary = root/"fixture-worker"
            binary.write_text("#!"+sys.executable+'''\nimport json,sys
if sys.argv[1] == '--preflight':
 print(json.dumps(dict(status='ready',device_count=2,p2p_0_to_1=1,p2p_1_to_0=1)))
else:
 rank=int(sys.argv[1]); mode=sys.argv[9]
 for seq in range(int(sys.argv[8])+1,int(sys.argv[8])+int(sys.argv[7])+1):
  print(json.dumps(dict(kind='sample',rank=rank,sequence=seq,mode=mode,bytes=int(sys.argv[4]),chunk_bytes=int(sys.argv[5]),work=int(sys.argv[6]),correct=True,checksum=123,wall_us=(20 if mode=='serial' else 10)+rank,gpu_us=9,process_cpu_us=12)))
 print(json.dumps(dict(kind='result',rank=rank,status='pass',transport='CudaIpc',channel='PortChannel')))
''')
            binary.chmod(0o755)
            script = Path(__file__).resolve().parents[1]/"run.py"
            result = subprocess.run([sys.executable, str(script), "--binary", str(binary), "--out", str(root/"out"),
                                     "--sizes", "4096", "--work", "0", "--iterations", "1", "--warmup", "0"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root/"out/summary.json").read_text())
            self.assertFalse(report["qualification"])
            self.assertEqual(len(report["performance"]), 2)
            for row in report["performance"]:
                self.assertEqual(row["repeat_pairs"], 7)
                self.assertEqual(row["region"], "beneficial")


if __name__ == "__main__": unittest.main()
