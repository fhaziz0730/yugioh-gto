from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Deck:
    main: list[int]
    extra: list[int]
    side: list[int]

    @classmethod
    def read(cls, path):
        sections = {"main": [], "extra": [], "side": []}
        section = None
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line in ("#main", "#extra", "!side"):
                section = line.lstrip("#!")
            elif line and not line.startswith(("#", "!")):
                if section is None or not line.isdigit() or int(line) <= 0:
                    raise ValueError(f"Invalid YDK line: {line!r}")
                sections[section].append(int(line))
        return cls(**sections)

    @property
    def active(self):
        return self.main + self.extra

    def errors(self):
        errors = []
        if not 40 <= len(self.main) <= 60:
            errors.append("Main Deck must contain 40–60 cards")
        if len(self.extra) > 15 or len(self.side) > 15:
            errors.append("Extra and Side Decks may contain at most 15 cards each")
        for code, count in Counter(self.active).items():
            if count > 3:
                errors.append(f"{code}: {count} copies in active deck (maximum 3)")
        return errors

    def write_active(self, path):
        text = "#main\n" + "\n".join(map(str, self.main))
        text += "\n#extra\n" + "\n".join(map(str, self.extra)) + "\n!side\n"
        Path(path).write_text(text, encoding="utf-8")


def supported_codes(path):
    return {int(line.split()[0]) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith("#")}
