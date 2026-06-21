"""Text-See-Do: a reproducible text-to-motion pipeline.

Components
----------
- tokenizer  : FSQ motion tokenizer (36-dim motion <-> discrete tokens)
- generator  : flan-t5 language model over motion tokens (multitask t2m/m2t/pred)
- verifiers  : dynamic (physical plausibility reward) + semantic (text-motion match)
- pipeline   : generate N candidates -> score -> best-of-N selection (inference)

Path resolution lives in :mod:`textseedo.paths`; the 36-dim motion layout lives in
:mod:`textseedo.motion_format`.
"""

__version__ = "0.1.0"
