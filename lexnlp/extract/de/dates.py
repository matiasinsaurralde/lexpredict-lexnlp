__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import os
from datetime import datetime
from typing import Generator, List, Optional

import joblib

from lexnlp.extract.all_locales.languages import Locale
from lexnlp.extract.common.annotations.date_annotation import DateAnnotation
from lexnlp.extract.de.date_model import DATE_MODEL_CHARS, DE_ALPHA_CHAR_SET

# Setup path
from lexnlp.extract.de.de_date_parser import DeDateParser


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# Load model lazily to avoid import-time errors with scikit-learn version mismatches
_MODEL_DATE = None

def _get_model_date():
    global _MODEL_DATE
    if _MODEL_DATE is None:
        try:
            _MODEL_DATE = joblib.load(os.path.join(MODULE_PATH, "./date_model.pickle"))
        except (ValueError, TypeError) as e:
            # Handle scikit-learn version mismatch - model may not be compatible
            import warnings
            warnings.warn(
                f"Could not load German date model due to scikit-learn version mismatch: {e}. "
                "German date extraction will work without the ML classifier.",
                UserWarning
            )
            _MODEL_DATE = None
    return _MODEL_DATE


parser = DeDateParser(DATE_MODEL_CHARS,
                      enable_classifier_check=True,
                      locale=Locale('de-DE'),
                      dateparser_settings={'PREFER_DAY_OF_MONTH': 'first',
                                           'STRICT_PARSING': False,
                                           'DATE_ORDER': 'DMY'},
                      classifier_model=None,  # Will be set lazily if available
                      alphabet_character_set=DE_ALPHA_CHAR_SET,
                      count_words=True,
                      feature_window=0)

# Set model lazily if available
_model = _get_model_date()
if _model is not None:
    parser.classifier_model = _model


get_dates = parser.get_dates

get_date_list = parser.get_date_list

get_date_annotations = parser.get_date_annotations

get_date_annotation_list = parser.get_date_annotation_list
