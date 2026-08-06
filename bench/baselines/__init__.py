"""Auto-discovered baseline package.

The runner scans this directory for every ``*.py`` whose filename does NOT start with ``_``
and expects each such module to expose two names:

    NAME  : str                    unique display name for the results table
    build(device) -> model         factory returning a model that obeys the interface
                                   contract in ``bench/interface.py``.

Copy ``_template.py`` to ``<method>.py`` to add a baseline. Files starting with ``_`` are
never auto-discovered (so ``_template.py`` is a spec, not a runnable baseline).
"""
