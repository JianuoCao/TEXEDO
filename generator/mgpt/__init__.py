"""mgpt package.

Backward-compat alias: checkpoints in this release were trained when the package was
named ``mGPT``. Their pickles reference ``mGPT.*`` modules/classes. We install a meta-path
finder so any ``mGPT`` / ``mGPT.<sub>`` import transparently resolves to ``mgpt.<sub>``,
letting those checkpoints load unchanged.
"""

import importlib
import importlib.abc
import importlib.util
import sys

_OLD = "mGPT"
_NEW = "mgpt"


class _MgptAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _OLD or fullname.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        new_name = _NEW + spec.name[len(_OLD):]
        module = importlib.import_module(new_name)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        pass


if not any(isinstance(f, _MgptAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _MgptAliasFinder())
