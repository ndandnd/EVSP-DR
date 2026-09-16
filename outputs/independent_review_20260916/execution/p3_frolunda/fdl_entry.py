"""Point unchanged pinned event CG at frozen FDL data, without dirtying its checkout."""
import sys
from pathlib import Path
code,data=map(Path,sys.argv[1:3]);sys.path.insert(0,str(code/'src'))
import exact_pricer_expanded as exact
exact.DATA_DIR=data.resolve()
raise SystemExit(exact.main(sys.argv[3:]))
