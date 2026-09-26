# sitecustomize.py - bind-mounted over /usr/lib/python3.12/sitecustomize.py
# Keeps Ubuntu's apport hook, then installs the MegaMoE hot / cold-tier hook.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import os
import sys

if os.environ.get("MEGA_HOOK"):
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("mega_peer_hook", os.environ["MEGA_HOOK"])
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.install()
    except Exception as _e:
        sys.stderr.write(f"MEGA_PEER hook FAILED to install: {_e!r}\n")
