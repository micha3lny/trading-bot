from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo


US_EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EquitySession:
    session_date: date
    open_utc: datetime | None
    close_utc: datetime | None
    is_early_close: bool
    is_trading_day: bool
    reason: str = ""


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    cur = date(year, month, 1)
    while cur.weekday() != weekday:
        cur += timedelta(days=1)
    return cur + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cur = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cur = date(year, month + 1, 1) - timedelta(days=1)
    while cur.weekday() != weekday:
        cur -= timedelta(days=1)
    return cur


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    actual = date(year, month, day)
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def _easter_date(year: int) -> date:
    # Anonymous Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def us_equity_market_holidays(year: int) -> dict[date, str]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1): "new_years_day",
        _nth_weekday(year, 1, 0, 3): "martin_luther_king_jr_day",
        _nth_weekday(year, 2, 0, 3): "presidents_day",
        _easter_date(year) - timedelta(days=2): "good_friday",
        _last_weekday(year, 5, 0): "memorial_day",
        _observed_fixed_holiday(year, 6, 19): "juneteenth",
        _observed_fixed_holiday(year, 7, 4): "independence_day",
        _nth_weekday(year, 9, 0, 1): "labor_day",
        _nth_weekday(year, 11, 3, 4): "thanksgiving_day",
        _observed_fixed_holiday(year, 12, 25): "christmas_day",
    }
    # Include adjacent observed New Year's Day when Jan 1 of next year lands on Saturday.
    next_new_year_observed = _observed_fixed_holiday(year + 1, 1, 1)
    if next_new_year_observed.year == year:
        holidays[next_new_year_observed] = "new_years_day_observed"
    return holidays


def us_equity_early_closes(year: int) -> dict[date, str]:
    early = {
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1): "day_after_thanksgiving",
    }
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 5 and christmas_eve not in us_equity_market_holidays(year):
        early[christmas_eve] = "christmas_eve"
    july_3 = date(year, 7, 3)
    if july_3.weekday() < 5 and july_3 not in us_equity_market_holidays(year):
        early[july_3] = "independence_day_eve"
    return early


def is_us_equity_trading_day(value: date | datetime) -> bool:
    session_date = value.date() if isinstance(value, datetime) else value
    return session_date.weekday() < 5 and session_date not in us_equity_market_holidays(session_date.year)


def get_us_equity_session(value: date | datetime) -> EquitySession:
    session_date = value.date() if isinstance(value, datetime) else value
    if session_date.weekday() >= 5:
        return EquitySession(session_date, None, None, False, False, "weekend")
    holiday_reason = us_equity_market_holidays(session_date.year).get(session_date)
    if holiday_reason:
        return EquitySession(session_date, None, None, False, False, holiday_reason)

    early_reason = us_equity_early_closes(session_date.year).get(session_date)
    open_local = datetime.combine(session_date, dtime(hour=9, minute=30), tzinfo=US_EASTERN)
    close_local_time = dtime(hour=13, minute=0) if early_reason else dtime(hour=16, minute=0)
    close_local = datetime.combine(session_date, close_local_time, tzinfo=US_EASTERN)
    open_utc = open_local.astimezone(timezone.utc)
    close_utc = close_local.astimezone(timezone.utc)
    return EquitySession(
        session_date=session_date,
        open_utc=open_utc,
        close_utc=close_utc,
        is_early_close=bool(early_reason),
        is_trading_day=True,
        reason=early_reason or "regular",
    )


def previous_us_equity_trading_day(value: date | datetime) -> date:
    cur = value.date() if isinstance(value, datetime) else value
    cur -= timedelta(days=1)
    while not is_us_equity_trading_day(cur):
        cur -= timedelta(days=1)
    return cur
