import os

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
_match = joblib.load(os.path.join(BASE, "models", "match_model.joblib"))
_player = joblib.load(os.path.join(BASE, "models", "player_model.joblib"))

_snap = _player["snap"]
_FEATS = _player["feats"]
DATA_THROUGH = _snap["LastDate"].max().date()
TOP_PLAYER_METRIC = "predicted fantasy points"

_recent = _snap[_snap["LastDate"] >= _snap["LastDate"].max() - pd.Timedelta(days=400)]
TEAMS = sorted(set(_recent["Team"]) & set(_match["elo"])) or sorted(_match["elo"])
VENUES = list(_match["venues"])


def _check(team: str) -> None:
    if team not in _match["elo"]:
        raise ValueError(f"Unknown team: {team}")


def predict_match_proba(home, away, match_date, venue=None) -> float:
    """Probability that the HOME team wins."""
    _check(home)
    _check(away)
    hf, hm = _match["form"][home]
    af, am = _match["form"][away]
    venue = venue if venue in VENUES else ("MCG" if "MCG" in VENUES else "Other")
    X = pd.DataFrame([dict(
        EloDiff=_match["elo"][home] - _match["elo"][away],
        FormDiff=hf - af, MarginDiff=hm - am, RestDiff=0, IsFinal=0,
        VenueGroup=venue)])
    return float(_match["model"].predict_proba(X)[0, 1])


def _team_scores(team, is_home):
    t = _snap[_snap["Team"] == team]
    if t.empty:
        raise ValueError(f"No player data for {team}")
    pool = t[t["LastDate"] >= t["LastDate"].max() - pd.Timedelta(days=35)].copy()  
    pool["IsHome"] = int(is_home)
    pool["Pred"] = _player["fantasy"].predict(pool[_FEATS])
    return [(f"{r.PlayerName} ({team})", float(r.Pred)) for r in pool.itertuples()]


def predict_top_player_scores(home, away, match_date, venue=None):
    """[(player, predicted fantasy score)] for both squads."""
    return _team_scores(home, True) + _team_scores(away, False)