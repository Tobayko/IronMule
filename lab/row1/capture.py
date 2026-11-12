"""Round 4 part 3: capture real inputs of named matmuls (ROW1_DUMPDIR/ROW1_DUMPNAMES set by the caller) over one
512-token chunk of the fixed text, stock path."""
import sys
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine
from ironmule.numeric_choice import TEXT
eng = GgmlEngine(sys.argv[1], speculate=False, seqs=1, seq_ctx=1024)
ids = eng.tokenize(TEXT.read_text())[1000:1513]
eng.chunk_nll(ids, [])
