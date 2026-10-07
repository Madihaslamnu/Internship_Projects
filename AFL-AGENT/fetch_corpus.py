import os
import re
import time

import requests

TITLES = [
    # rules and competition
    "Australian rules football",
    "Laws of Australian football",
    "Australian Football League",
    "AFL Grand Final",
    "History of Australian rules football",
    "Brownlow Medal",
    "Coleman Medal",
    # clubs
    "Collingwood Football Club",
    "Carlton Football Club",
    "Essendon Football Club",
    "Richmond Football Club",
    "Geelong Football Club",
    "Hawthorn Football Club",
    "Sydney Swans",
    "Brisbane Lions",
    "West Coast Eagles",
    "Melbourne Football Club",
    "Western Bulldogs",
    "North Melbourne Football Club",
    "Adelaide Football Club",
    "Port Adelaide Football Club",
    "Fremantle Football Club",
    "St Kilda Football Club",
    "Gold Coast Suns",
    "Greater Western Sydney Giants",
    # recent seasons
    "2021 AFL season",
    "2022 AFL season",
    "2023 AFL season",
    "2024 AFL season",
    "2025 AFL season",
]

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "AFL-RAG-student-project/1.0 (educational use)"}
os.makedirs("data", exist_ok=True)

for title in TITLES:
    fname = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".md"
    path = os.path.join("data", fname)
    if os.path.exists(path):
        print(f"exists, skipping: {fname}")
        continue
    params = {"action": "query", "prop": "extracts", "explaintext": 1,
              "redirects": 1, "titles": title, "format": "json"}
    for attempt in range(4):
        resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 200 and resp.text.lstrip().startswith("{"):
            break
        time.sleep(5 * (attempt + 1))
    else:
        print(f"FAILED after retries: {title}")
        continue
    page = next(iter(resp.json()["query"]["pages"].values()))
    text = page.get("extract", "")
    if not text:
        print(f"SKIP (not found): {title}")
        continue
    text = re.sub(r"^===\s*(.+?)\s*===$", r"### \1", text, flags=re.M)
    text = re.sub(r"^==\s*(.+?)\s*==$", r"## \1", text, flags=re.M)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {page['title']}\n\nSource: Wikipedia, CC BY-SA 4.0\n\n{text}")
    print(f"saved {path} ({len(text):,} chars)")
    time.sleep(2)
    