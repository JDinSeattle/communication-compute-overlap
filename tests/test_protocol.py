import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import model
from run import validate_samples,region


class ProtocolTests(unittest.TestCase):
    def test_ordered_protocol_exhaustion(self):
        for n in (1,2,3,8):
            r=model.explore(n);self.assertTrue(r["safe"]);self.assertEqual(r["terminal_states"],1)

    def test_missing_ack_has_overwrite_witness(self):
        r=model.explore(require_ack=False)
        self.assertFalse(r["safe"]);self.assertIn("generation 2",r["witness"][-1])

    def test_early_signal_has_stale_witness(self):
        r=model.explore(ordered_signal=False)
        self.assertFalse(r["safe"]);self.assertIn("generation 0",r["witness"][-1])

    @staticmethod
    def logs():
        return ["\n".join(json.dumps(r) for r in [
            {"kind":"sample","rank":rank,"sequence":1,"mode":"overlap","bytes":4096,"chunk_bytes":16384,"work":32,"correct":True,"checksum":123,"wall_us":10+rank*4,"gpu_us":9,"process_cpu_us":12},
            {"kind":"result","rank":rank,"status":"pass","transport":"CudaIpc","channel":"PortChannel"}]) for rank in (0,1)]

    def test_slower_rank_is_critical_path(self):
        self.assertEqual(validate_samples(self.logs(),1,0,"overlap",4096,16384,32),[14])

    def test_correctness_precedes_performance(self):
        logs=self.logs();logs[1]=logs[1].replace('"correct": true','"correct": false')
        with self.assertRaises(ValueError):validate_samples(logs,1,0,"overlap",4096,16384,32)

    def test_missing_duplicate_or_bad_timing_rejected(self):
        for change in (lambda s:s.splitlines()[0],lambda s:s+'\n'+s.splitlines()[0],lambda s:s.replace('"wall_us": 10','"wall_us": -1')):
            logs=self.logs();logs[0]=change(logs[0])
            with self.assertRaises(ValueError):validate_samples(logs,1,0,"overlap",4096,16384,32)

    def test_benefit_regions(self):
        self.assertEqual(region(1.2),"beneficial");self.assertEqual(region(1.01),"no_material_gain");self.assertEqual(region(.8),"regressed")
        with self.assertRaises(ValueError):region(float("nan"))


if __name__=="__main__":unittest.main()
