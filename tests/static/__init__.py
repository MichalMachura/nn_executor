import os, sys
__path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if __path not in sys.path:
    sys.path.append(__path)

import test_analyzer