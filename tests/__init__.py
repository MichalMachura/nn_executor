import os, sys
__path = os.path.abspath(os.path.dirname(__file__))
if __path not in sys.path:
    sys.path.append(__path)

import shared_tests_data
import static
import test_models
import test_modifiers
import test_parse
import test_prune
import test_utils

print("xd")
