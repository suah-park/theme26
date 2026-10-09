import json
import sys
from pathlib import Path
import numpy as np
from .config import LBMConfig
from .engine import run_case
from .reporting import write_json

def main():
    request=json.loads(Path(sys.argv[1]).read_text())
    pore=np.load(request['input'],allow_pickle=False)
    result=run_case(pore,tuple(request['spacing']),LBMConfig(**request['config']))
    write_json(request['output'],result.as_dict())

if __name__=='__main__': main()
