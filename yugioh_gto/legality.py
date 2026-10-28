"""Fail-closed, dated TCG eligibility snapshots; unknown release dates are rejected."""
from collections import Counter
from datetime import date
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import Request, urlopen

BANLIST = "https://www.yugioh-card.com/en/limited/list_2026-10-19/"
CARDS = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"


def normalize(name):
    return " ".join(unicodedata.normalize("NFKC", name).upper().split())


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            if self.row is not None:
                self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "yugioh-gto-research/0.1"}), timeout=90) as r:
        return r.read().decode("utf-8")


def refresh(output, as_of, banlist_url=BANLIST):
    """Download evidence, retain provenance, never infer release from script presence."""
    date.fromisoformat(as_of)
    html = fetch(banlist_url)
    effective = re.search(r"Effective\s+from\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", html)
    if not effective:
        raise ValueError("Could not identify official banlist effective date")
    from datetime import datetime
    effective_date = datetime.strptime(effective[1], "%B %d, %Y").date().isoformat()
    if effective_date > as_of:
        raise ValueError("Selected banlist is not yet effective")
    parser = Tables()
    parser.feed(html)
    limits = {}
    for row in parser.rows:
        if len(row) >= 3:
            status = row[2].lower().strip()
            if status in ("forbidden", "limited", "semi-limited"):
                limits[normalize(row[1])] = {"forbidden": 0, "limited": 1, "semi-limited": 2}[status]
    if len(limits) < 50:
        raise ValueError("Banlist parsing failed sanity check; refusing partial snapshot")
    cards = {}
    raw = json.loads(fetch(CARDS))
    for card in raw["data"]:
        info = card.get("misc_info", [{}])[0]
        released = info.get("tcg_date")
        entry = {"name": card["name"], "tcg_release": released,
                 "limit": limits.get(normalize(card["name"]), 3),
                 "type": card.get("type", ""), "alias": card["id"]}
        # Alternate artwork passwords count towards the same card limit.
        for code in {card["id"], *(im["id"] for im in card.get("card_images", []))}:
            cards[str(code)] = entry
    snapshot = {"as_of": as_of, "effective": effective_date,
                "banlist_url": banlist_url, "cards_url": CARDS,
                "release_source": "YGOPRODeck misc_info.tcg_date (secondary source)",
                "cards": cards}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    output.with_suffix(".source.html").write_text(html, encoding="utf-8")
    return snapshot


def validate(deck, snapshot, as_of):
    errors = deck.errors()
    if snapshot.get("as_of") != as_of:
        errors.append(f"Snapshot date must equal experiment date {as_of}")
    if snapshot.get("effective", "9999") > as_of:
        errors.append("Banlist is not yet effective")
    counts = Counter()
    entries = {}
    for code in deck.active:
        card = snapshot["cards"].get(str(code))
        if card is None:
            errors.append(f"Unknown card password: {code}")
            continue
        alias = card.get("alias", code)
        counts[alias] += 1
        entries[alias] = card
        released = card.get("tcg_release")
        if not released or released > as_of:
            errors.append(f"{card['name']} ({code}): TCG release missing or later than {as_of}")
    for alias, count in counts.items():
        card = entries[alias]
        if count > card["limit"]:
            errors.append(f"{card['name']}: {count} copies; TCG limit is {card['limit']}")
    extra_types = ("Fusion", "Synchro", "Xyz", "Link")
    for section, codes in (("main", deck.main), ("extra", deck.extra)):
        for code in set(codes):
            card = snapshot["cards"].get(str(code))
            if card and card.get("type"):
                is_extra = any(t.lower() in card["type"].lower() for t in extra_types)
                if is_extra != (section == "extra"):
                    errors.append(f"{card['name']}: invalid {section} deck placement")
    return sorted(set(errors))
