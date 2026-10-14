"""Element anchors for dynamic-page macro replay (schema v3).

A coordinate-only macro breaks as soon as a list reorders: the recorded
``(x, y)`` hits the wrong element and the ``pre_state`` similarity gate
fails. An anchor pins the *identity* of the tapped element
(resource_id / text / class) at record time so replay can re-resolve the
element's *current* coordinates before tapping.

Resolution order (first hit wins):
1. ``resource_id`` exact match (optionally tightened by text).
2. ``text`` + ``class`` match.
3. ``text``-only match.
4. ``path`` positional heuristic (same index path, same screen size).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

_ANCHOR_SCHEMA_VERSION = "3.0"


def _first_present(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _norm_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def build_anchor(element: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build an identity anchor from a raw UI element dict.

    Returns ``None`` when the element carries no identifying field, in
    which case the action stays coordinate-only.
    """
    if not isinstance(element, dict):
        return None
    anchor = {
        "resource_id": _first_present(
            element, "resourceId", "resource_id", "viewIdResourceName", "id"
        ),
        "text": _norm_text(_first_present(element, "text", "label", "value")),
        "class": _first_present(element, "className", "class_name", "class", "type"),
        "content_description": _norm_text(
            _first_present(
                element,
                "contentDescription",
                "content_desc",
                "content_description",
                "description",
            )
        ),
    }
    if not any(v for v in anchor.values()):
        return None
    return anchor


def _node_center(node: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    bounds = node.get("bounds") or {}
    if "center_x" in bounds and "center_y" in bounds:
        return int(bounds["center_x"]), int(bounds["center_y"])
    try:
        left, top, right, bottom = (
            int(bounds["left"]),
            int(bounds["top"]),
            int(bounds["right"]),
            int(bounds["bottom"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return (left + right) // 2, (top + bottom) // 2


def resolve_anchor(
    anchor: Dict[str, Any], nodes: List[Dict[str, Any]]
) -> Optional[Tuple[Tuple[int, int], str]]:
    """Resolve an anchor against current normalized nodes.

    Returns ``((x, y), strategy)`` or ``None`` when nothing matches.
    ``strategy`` is one of ``resource_id`` / ``text+class`` / ``text`` /
    ``path`` and is recorded in the step result for later analysis.
    """
    if not anchor or not nodes:
        return None

    rid = anchor.get("resource_id")
    text = anchor.get("text")
    cls = anchor.get("class")
    desc = anchor.get("content_description")

    # 1. resource_id exact (tighten by text when the anchor has one).
    if rid:
        for node in nodes:
            if node.get("resource_id") != rid:
                continue
            if text and node.get("text") != text:
                continue
            center = _node_center(node)
            if center:
                return center, "resource_id"

    # 2. text + class.
    if text and cls:
        for node in nodes:
            if node.get("text") == text and node.get("class") == cls:
                center = _node_center(node)
                if center:
                    return center, "text+class"

    # 3. text-only (also covers content_description stored as text).
    if text:
        for node in nodes:
            if node.get("text") == text:
                center = _node_center(node)
                if center:
                    return center, "text"
    if desc and desc != text:
        for node in nodes:
            if node.get("content_description") == desc:
                center = _node_center(node)
                if center:
                    return center, "content_description"

    # 4. path heuristic is resolved by the caller (needs the recorded
    #    pre_state screen size); see resolve_anchor_with_screen.
    return None


def resolve_anchor_with_screen(
    anchor: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    recorded_screen: Optional[Dict[str, Any]],
    current_screen: Optional[Dict[str, Any]],
) -> Optional[Tuple[Tuple[int, int], str]]:
    """Resolve with the path heuristic as a last resort.

    Only applies when the screen size is unchanged (same device /
    orientation); otherwise positional matching is meaningless.
    """
    hit = resolve_anchor(anchor, nodes)
    if hit:
        return hit
    path = anchor.get("path")
    if not path or not recorded_screen or not current_screen:
        return None
    if recorded_screen.get("width") != current_screen.get(
        "width"
    ) or recorded_screen.get("height") != current_screen.get("height"):
        return None
    wanted = tuple(path)
    for node in nodes:
        node_path = node.get("path")
        if node_path is not None and tuple(node_path) == wanted:
            center = _node_center(node)
            if center:
                return center, "path"
    return None
