"""What `update\\windows.exe` starts (NN-203). That file is the app's launcher, the same bytes as `windows\\SiLVANN.exe`,
so the updater needs no signature of its own: the launcher starts `python\\pythonw.exe` on
`scripts\\portable\\silvann_portable.py`, both beside itself. In `update\\` the file of that name is this one
(▶ pack_portable.py), and it runs the updater, silvann_update.py, beside it."""
import runpy
import sys
from pathlib import Path

updater = Path(__file__).resolve().parent / "silvann_update.py"
sys.argv = [str(updater)] + sys.argv[1:]
runpy.run_path(str(updater), run_name="__main__")
