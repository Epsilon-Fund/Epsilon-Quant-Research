from __future__ import annotations

import re
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Static template list — everything except "-on-{month}-{day}-{year}"
# Edit this list to add or remove cities / lowest markets.
# ---------------------------------------------------------------------------

SLUG_TEMPLATES: tuple[str, ...] = (
    "highest-temperature-in-hong-kong",
    "highest-temperature-in-tokyo",
    "highest-temperature-in-wellington",
    "highest-temperature-in-seoul",
    "highest-temperature-in-london",
    "highest-temperature-in-madrid",
    "highest-temperature-in-lucknow",
    "highest-temperature-in-shanghai",
    "highest-temperature-in-paris",
    "highest-temperature-in-beijing",
    "highest-temperature-in-austin",
    "highest-temperature-in-moscow",
    "highest-temperature-in-singapore",
    "highest-temperature-in-nyc",
    "highest-temperature-in-chengdu",
    "highest-temperature-in-houston",
    "highest-temperature-in-chicago",
    "highest-temperature-in-jakarta",
    "highest-temperature-in-warsaw",
    "highest-temperature-in-wuhan",
    "lowest-temperature-in-miami",
    "highest-temperature-in-munich",
    "highest-temperature-in-atlanta",
    "highest-temperature-in-taipei",
    "highest-temperature-in-shenzhen",
    "highest-temperature-in-buenos-aires",
    "lowest-temperature-in-tokyo",
    "highest-temperature-in-mexico-city",
    "highest-temperature-in-miami",
    "highest-temperature-in-dallas",
    "highest-temperature-in-tel-aviv",
    "highest-temperature-in-amsterdam",
    "highest-temperature-in-chongqing",
    "highest-temperature-in-seattle",
    "highest-temperature-in-lagos",
    "highest-temperature-in-toronto",
    "highest-temperature-in-milan",
    "highest-temperature-in-los-angeles",
    "highest-temperature-in-san-francisco",
    "highest-temperature-in-sao-paulo",
    "highest-temperature-in-istanbul",
    "highest-temperature-in-ankara",
    "highest-temperature-in-denver",
    "highest-temperature-in-helsinki",
    "lowest-temperature-in-nyc",
    "highest-temperature-in-busan",
    "lowest-temperature-in-seoul",
    "highest-temperature-in-guangzhou",
    "highest-temperature-in-manila",
    "highest-temperature-in-jeddah",
    "highest-temperature-in-panama-city",
    "lowest-temperature-in-shanghai",
    "highest-temperature-in-kuala-lumpur",
    "highest-temperature-in-karachi",
    "lowest-temperature-in-hong-kong",
    "lowest-temperature-in-london",
    "highest-temperature-in-cape-town",
    "highest-temperature-in-qingdao",
    "lowest-temperature-in-paris",
)

# Regex that matches the "-on-{month}-{day}-{year}" date suffix.
_DATE_SUFFIX_RE = re.compile(r"-on-([a-z]+)-(\d{1,2})-(\d{4})$")

_MONTH_NAMES = (
    "", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def make_slug(template: str, for_date: date) -> str:
    """Append a date suffix to a city template.

    Example: make_slug("highest-temperature-in-tokyo", date(2026, 5, 7))
             → "highest-temperature-in-tokyo-on-may-7-2026"
    """
    month = _MONTH_NAMES[for_date.month]
    return f"{template}-on-{month}-{for_date.day}-{for_date.year}"


def template_for_slug(slug: str) -> str | None:
    """Strip the date suffix and return the base template, or None if not a weather slug."""
    m = _DATE_SUFFIX_RE.search(slug)
    if m is None:
        return None
    return slug[: m.start()]


def date_for_slug(slug: str) -> date | None:
    """Parse the date embedded in a slug, or None if not a weather slug."""
    m = _DATE_SUFFIX_RE.search(slug)
    if m is None:
        return None
    month_str, day_str, year_str = m.groups()
    try:
        month = _MONTH_NAMES.index(month_str.lower())
        return date(int(year_str), month, int(day_str))
    except (ValueError, IndexError):
        return None


def next_day_slug(slug: str) -> str | None:
    """Return the same market's slug for the following calendar day."""
    template = template_for_slug(slug)
    current_date = date_for_slug(slug)
    if template is None or current_date is None:
        return None
    return make_slug(template, current_date + timedelta(days=1))


def generate_slugs_for_date(for_date: date) -> tuple[str, ...]:
    """Generate all 59 weather slugs for a given date."""
    return tuple(make_slug(t, for_date) for t in SLUG_TEMPLATES)


def generate_today_slugs() -> tuple[str, ...]:
    """Generate all weather slugs for today's UTC date."""
    from datetime import datetime, timezone
    return generate_slugs_for_date(datetime.now(timezone.utc).date())
