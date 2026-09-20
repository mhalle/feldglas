"""Importing feldglas pulls nothing heavy - the rule the rest of the family keeps too. The
package is what a viewer, a notebook or a server's API container imports to pool a gate in
0.2 ms; it must not drag a GPU stack in behind it."""
import subprocess, sys, unittest


class LightImport(unittest.TestCase):
    def test_import_pulls_no_torch_scipy_modal_or_store_client(self):
        code = ("import sys, feldglas, feldglas.gate, feldglas.store, feldglas.observe, feldglas.heads, "
                "feldglas.suite, feldglas.adapters.radar, feldglas.adapters.null, feldglas.remote\n"
                "bad = [m for m in ('torch', 'scipy', 'modal', 'sklearn', 'zarr', 'obstore', 'provender') if m in sys.modules]\n"
                "assert not bad, bad")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
