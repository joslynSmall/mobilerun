"""Mobilerun Macro Module - Record and replay UI automation sequences."""

__all__ = [
    "MacroPlayer",
    "replay_macro_file",
    "replay_macro_folder",
    "build_anchor",
    "resolve_anchor",
    "resolve_anchor_with_screen",
    "load_cases",
    "record_case",
    "find_matching_cases",
    "format_case_summary",
]

_LAZY_MODULES = {
    "MacroPlayer": "replay",
    "replay_macro_file": "replay",
    "replay_macro_folder": "replay",
    "build_anchor": "anchor",
    "resolve_anchor": "anchor",
    "resolve_anchor_with_screen": "anchor",
    "load_cases": "resolutions",
    "record_case": "resolutions",
    "find_matching_cases": "resolutions",
    "format_case_summary": "resolutions",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        from importlib import import_module

        module = import_module(f"mobilerun.macro.{_LAZY_MODULES[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
