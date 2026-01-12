"""Date extraction for English.

This module implements date extraction functionality in English.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


# pylint: disable=bare-except

# Standard imports
import os
import string

# Setup path


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# Load model lazily to avoid import-time errors with scikit-learn version mismatches
_MODEL_DATE = None

def _get_model_date():
    global _MODEL_DATE
    if _MODEL_DATE is None:
        from lexnlp.utils.unpickler import safe_joblib_load
        _MODEL_DATE = safe_joblib_load(os.path.join(MODULE_PATH, "./date_model.pickle"))
        if _MODEL_DATE is None:
            raise RuntimeError("Date model could not be loaded due to scikit-learn version mismatch")
    return _MODEL_DATE

# Expose MODEL_DATE as a proxy that loads lazily
class _ModelDateProxy:
    def __getattr__(self, name):
        return getattr(_get_model_date(), name)
    
    def __call__(self, *args, **kwargs):
        return _get_model_date()(*args, **kwargs)
    
    def __getitem__(self, key):
        return _get_model_date()[key]
    
    def __iter__(self):
        return iter(_get_model_date())
    
    def __len__(self):
        return len(_get_model_date())

MODEL_DATE = _ModelDateProxy()

ALPHA_CHAR_SET = set(string.ascii_letters)
DATE_MODEL_CHARS = []
DATE_MODEL_CHARS.extend(string.ascii_letters)
DATE_MODEL_CHARS.extend(string.digits)
DATE_MODEL_CHARS.extend(["-", "/", " ", "%", "#", "$"])
