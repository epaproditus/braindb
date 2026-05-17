"""
Fuzzy + full-text search against the entities table.
Uses a 4-tier scoring system:
  1. AND tsquery match (all words) — weight 1.0
  2. OR tsquery match (any word)  — weight 0.3
  3. Content trigram similarity    — weight 0.5
  4. Title trigram similarity      — weight 0.3
"""
import os

import psycopg2.extras

# ------------------------------------------------------------------ #
# Central content-preview helper (shared by recall/search/list/etc.)  #
# ------------------------------------------------------------------ #
# Lives here because search.py is a dependency-free leaf module that
# context.py and the agent tools already import — so this is reused, not
# a new module. The ONLY full-content read is get_entity(<id>); every
# multi-item path renders previews so big/polluted bodies never flood
# (or pollute) the caller's context.
PREVIEW_CAP = int(os.getenv("BRAINDB_PREVIEW_CAP", "1024"))  # <= 1K per item


def preview(text, entity_id=None, cap: int = PREVIEW_CAP) -> str:
    """Bound a content string to `cap` chars; if cut, append the standard
    marker + drill-down protocol so the LLM knows how to read the full body."""
    s = "" if text is None else str(text)
    if len(s) <= cap:
        return s
    extra = len(s) - cap
    how = f' full body: get_entity("{entity_id}").' if entity_id else "."
    return (
        s[:cap]
        + f"\n--truncated ({extra} more chars)--{how} If large, "
        "delegate_to_subagent to read/extract it without polluting this context."
    )


# Shared SQL fragments
_OR_TSQUERY = "to_tsquery('english', regexp_replace(plainto_tsquery('english', %s)::text, ' & ', ' | ', 'g'))"

_SCORE_EXPR = f"""
    COALESCE(
        CASE WHEN e.search_vector @@ plainto_tsquery('english', %s)
             THEN ts_rank(e.search_vector, plainto_tsquery('english', %s))
             ELSE 0 END, 0)
    + COALESCE(
        CASE WHEN e.search_vector @@ {_OR_TSQUERY}
             AND NOT (e.search_vector @@ plainto_tsquery('english', %s))
             THEN ts_rank(e.search_vector, {_OR_TSQUERY}) * 0.3
             ELSE 0 END, 0)
    + COALESCE(similarity(e.content, %s), 0) * 0.5
    + COALESCE(similarity(COALESCE(e.title, ''), %s), 0) * 0.3
    AS score
"""

_WHERE_EXPR = f"""
    WHERE (
        e.search_vector @@ plainto_tsquery('english', %s)
        OR e.search_vector @@ {_OR_TSQUERY}
        OR similarity(e.content, %s) > 0.15
        OR similarity(COALESCE(e.title, ''), %s) > 0.2
    )
"""


def fuzzy_search(conn, query: str, entity_types: list[str] | None, min_importance: float, limit: int) -> list[dict]:
    # Score: AND check + AND rank (2) + OR tsquery + NOT AND + OR tsquery rank (3) + trigram x2 = 7
    score_params = (query,) * 7
    # Where: AND + OR tsquery + trigram content + trigram title = 4
    where_params = (query,) * 4

    select = f"""
        SELECT
            e.id, e.entity_type, e.title, e.content, e.summary,
            e.keywords, e.importance, e.source, e.notes,
            e.created_at, e.updated_at, e.accessed_at, e.access_count, e.metadata,
            {_SCORE_EXPR}
        FROM entities e
        {_WHERE_EXPR}
    """

    if entity_types:
        sql = select + """
            AND e.entity_type = ANY(%s)
            AND e.importance >= %s
            ORDER BY score DESC
            LIMIT %s
        """
        params = score_params + where_params + (entity_types, min_importance, limit)
    else:
        sql = select + """
            AND e.importance >= %s
            ORDER BY score DESC
            LIMIT %s
        """
        params = score_params + where_params + (min_importance, limit)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
    # Central preview cap — covers /memory/search + quick_search (and the
    # text seeds feeding /memory/context). Real content is read only via
    # get_entity(<id>) (the full carve-out).
    for r in rows:
        r["content"] = preview(r.get("content"), r.get("id"))
    return rows
