import os
import pathlib

thisdir = pathlib.Path(__file__).parent.resolve()
datadir = thisdir / "data"
resultsdir = thisdir / "results"
outputdir = thisdir / "output"

num_processors = max(1, (os.cpu_count() or 1) - 3)

os.environ["SAGA_DATA_DIR"] = str(datadir)
