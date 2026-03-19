from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParsedLine:
    speaker: Optional[str]
    content: str


# Treat "SPEAKER: content" only when it looks like an actual dialogue label:
# - must be at start of line
# - label is short (avoid matching normal sentences like "시간: 12:30" too aggressively)
# - content must exist
_SPEAKER_LINE_RE = re.compile(r"^\s*([^\s:]{1,12})\s*:\s*(.+?)\s*$")


def parse_dialogue_line(raw_line: str) -> ParsedLine:
    """
    Parse one line into (speaker?, content).

    Philosophy note:
    - This is NOT "learning logic"; it's an input contract normalizer.
    - Only strips labels when the line matches a strict speaker-prefix form.
    """
    line = raw_line.strip()
    if not line:
        return ParsedLine(speaker=None, content="")

    m = _SPEAKER_LINE_RE.match(line)
    if m:
        speaker = m.group(1)
        content = m.group(2).strip()
        return ParsedLine(speaker=speaker, content=content)

    return ParsedLine(speaker=None, content=line)

