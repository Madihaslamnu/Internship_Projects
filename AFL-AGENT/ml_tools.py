import difflib
from datetime import date, datetime, timedelta
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ToolError(Exception):
    """A user-facing, safe-to-display tool failure."""


TEAMS = [
    "Adelaide", "Brisbane Lions", "Carlton", "Collingwood", "Essendon", "Fremantle",
    "Geelong", "Gold Coast", "GWS", "Hawthorn", "Melbourne", "North Melbourne",
    "Port Adelaide", "Richmond", "St Kilda", "Sydney", "West Coast", "Western Bulldogs",
]
ALIASES = {
    "pies": "Collingwood", "magpies": "Collingwood", "collingwood magpies": "Collingwood",
    "cats": "Geelong", "geelong cats": "Geelong", "swans": "Sydney", "sydney swans": "Sydney",
    "bombers": "Essendon", "blues": "Carlton", "tigers": "Richmond", "hawks": "Hawthorn",
    "dees": "Melbourne", "demons": "Melbourne", "saints": "St Kilda", "crows": "Adelaide",
    "adelaide crows": "Adelaide", "power": "Port Adelaide", "port": "Port Adelaide",
    "eagles": "West Coast", "west coast eagles": "West Coast", "dockers": "Fremantle",
    "freo": "Fremantle", "lions": "Brisbane Lions", "brisbane": "Brisbane Lions",
    "suns": "Gold Coast", "gold coast suns": "Gold Coast", "giants": "GWS",
    "gws giants": "GWS", "greater western sydney": "GWS",
    "greater western sydney giants": "GWS", "bulldogs": "Western Bulldogs",
    "dogs": "Western Bulldogs", "footscray": "Western Bulldogs", "roos": "North Melbourne",
    "kangaroos": "North Melbourne", "north": "North Melbourne",
}
MIN_DATE = date(2000, 1, 1) 
MAX_AHEAD_DAYS = 120         


def canonical_team(name: str) -> str:
    key = (name or "").strip().lower()
    for t in TEAMS:
        if key == t.lower():
            return t
    if key in ALIASES:
        return ALIASES[key]
    close = difflib.get_close_matches(key, [t.lower() for t in TEAMS], n=1, cutoff=0.75)
    if close:
        return next(t for t in TEAMS if t.lower() == close[0])
    raise ToolError(f"\"{name}\" is not a recognised AFL team. "
                    f"Valid teams are: {', '.join(TEAMS)}.")


def parse_date(text: Optional[str]) -> date:
    if not text:
        return date.today()
    try:
        d = datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ToolError(f"The date \"{text}\" isn't valid. Please use the format YYYY-MM-DD.")
    latest = date.today() + timedelta(days=MAX_AHEAD_DAYS)
    if d < MIN_DATE:
        raise ToolError(f"{d} is too far back. The model only covers matches from {MIN_DATE} onward.")
    if d > latest:
        raise ToolError(f"{d} is too far ahead. I can only predict matches up to {latest}.")
    return d


def _backend():
    try:
        import day2_models
        return day2_models
    except ImportError:
        raise ToolError("The prediction models aren't connected yet (day2_models.py not found), "
                        "so I can't make a real prediction.")


class MatchInput(BaseModel):
    home_team: str = Field(description="Home team name, e.g. 'Collingwood'")
    away_team: str = Field(description="Away team name, e.g. 'Carlton'")
    match_date: Optional[str] = Field(
        default=None, description="Match date as YYYY-MM-DD. Omit for today/the upcoming match.")
    venue: Optional[str] = Field(default=None, description="Venue name, e.g. 'MCG'. Optional.")


def _prepare(home_team, away_team, match_date):
    h, a = canonical_team(home_team), canonical_team(away_team)
    if h == a:
        raise ToolError("Home and away teams must be different.")
    return h, a, parse_date(match_date)


@tool("predict_match_winner", args_schema=MatchInput)
def predict_match_winner(home_team: str, away_team: str,
                         match_date: Optional[str] = None, venue: Optional[str] = None) -> dict:
    """Predict the winner of an AFL match with the trained match-winner model.
    Use for questions like 'who will win Collingwood vs Carlton?'."""
    h, a, d = _prepare(home_team, away_team, match_date)
    p_home = float(_backend().predict_match_proba(h, a, d, venue))
    if not 0.0 <= p_home <= 1.0:
        raise ToolError("The model returned an invalid probability.")
    return {
        "home_team": h, "away_team": a, "match_date": str(d), "venue": venue,
        "predicted_winner": h if p_home >= 0.5 else a,
        "home_win_probability": round(p_home, 3),
        "away_win_probability": round(1 - p_home, 3),
    }


@tool("predict_top_player", args_schema=MatchInput)
def predict_top_player(home_team: str, away_team: str,
                       match_date: Optional[str] = None, venue: Optional[str] = None) -> dict:
    """Predict the top-performing players in an AFL match with the trained player model.
    Use for 'who will be the best player / top disposal getter in X vs Y?'."""
    h, a, d = _prepare(home_team, away_team, match_date)
    backend = _backend()
    ranked = sorted(backend.predict_top_player_scores(h, a, d, venue),
                    key=lambda x: x[1], reverse=True)[:5]
    if not ranked:
        raise ToolError("The model returned no players for that match.")
    return {
        "home_team": h, "away_team": a, "match_date": str(d),
        "metric": getattr(backend, "TOP_PLAYER_METRIC", "model score"),
        "top_players": [{"player": p, "predicted_score": round(float(s), 2)} for p, s in ranked],
    }
    