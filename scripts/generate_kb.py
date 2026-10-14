"""Deterministic generator for the Part 1 knowledge base (Atlas AI Summit).

Re-running this script with the same SEED produces byte-identical
sessions.csv, speakers.json, and venue_guide.pdf. evals/kb_gold.jsonl is
derived entirely from the generated session/speaker/room data (never
hand-written).

Usage:
    uv run python scripts/generate_kb.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path

import pandas as pd
import reportlab.rl_config as rl_config
from faker import Faker
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Global invariant mode: makes reportlab skip non-deterministic debug
# comments and produce a stable /ID, so venue_guide.pdf is byte-identical
# across runs (on top of passing invariant=1 to the Canvas itself).
rl_config.invariant = 1

SEED = 42

#: Bumped deliberately whenever the generated KB content changes (see
#: README "Knowledge base > Conventions"). manifest.json pins the data to
#: this version via per-file hashes, so a content change without a version
#: bump is caught by tests/test_kb.py::test_manifest_hashes_match_files.
KB_VERSION = "1.0.2"

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"
EVALS_DIR = ROOT / "evals"

CONFERENCE_NAME = "Atlas AI Summit"
CONFERENCE_DATES = {1: "2026-11-18", 2: "2026-11-19"}
VENUE_CITY = "Johannesburg"

TRACKS = ["RAG", "Agents", "MLOps", "Safety", "Applied AI"]
LEVELS = ["Beginner", "Intermediate", "Advanced"]

ROOMS = [
    {
        "room_id": "R-01",
        "name": "Hall A",
        "floor": 1,
        "capacity": 300,
        "wheelchair_accessible": "Yes",
        "av_equipment": "Projector, PA system, lapel mics, live captioning",
        "hearing_loop": "Yes",
    },
    {
        "room_id": "R-02",
        "name": "Hall B",
        "floor": 1,
        "capacity": 250,
        "wheelchair_accessible": "Yes",
        "av_equipment": "Projector, PA system, lapel mics",
        "hearing_loop": "Yes",
    },
    {
        "room_id": "R-03",
        "name": "Workshop 1",
        "floor": 2,
        "capacity": 60,
        "wheelchair_accessible": "Yes",
        "av_equipment": "Projector, whiteboard, HDMI",
        "hearing_loop": "No",
    },
    {
        "room_id": "R-04",
        "name": "Workshop 2",
        "floor": 2,
        "capacity": 60,
        "wheelchair_accessible": "No",
        "av_equipment": "Projector, whiteboard",
        "hearing_loop": "No",
    },
    {
        "room_id": "R-05",
        "name": "Terrace Room",
        "floor": 2,
        "capacity": 80,
        "wheelchair_accessible": "Yes",
        "av_equipment": "Projector, PA system",
        "hearing_loop": "No",
    },
    {
        "room_id": "R-06",
        "name": "Boardroom",
        "floor": 3,
        "capacity": 30,
        "wheelchair_accessible": "Yes",
        "av_equipment": "TV screen, video conferencing",
        "hearing_loop": "No",
    },
]

SLOTS = [
    ("09:00", "10:00"),
    ("10:15", "11:15"),
    ("11:30", "12:30"),
    ("13:30", "14:30"),
    ("14:45", "15:45"),
    ("16:00", "17:00"),
]

TRACK_TITLES = {
    "RAG": [
        "Retrieval-Augmented Generation for Enterprise Search",
        "Chunking Strategies that Actually Work",
        "Hybrid Retrieval: BM25 Meets Embeddings",
        "Evaluating RAG Pipelines at Scale",
        "Query Routing for Multi-Source RAG",
        "Reducing Hallucinations with Grounded Retrieval",
        "Building a RAG System from Scratch",
        "Vector Databases: A Practical Comparison",
    ],
    "Agents": [
        "Designing Multi-Agent Workflows with LangGraph",
        "Tool Use and Function Calling in Practice",
        "Agentic Planning: Lessons from Production",
        "Memory Architectures for Long-Running Agents",
        "Human-in-the-Loop Agent Design",
        "Debugging Autonomous Agents",
        "Orchestrating Agents Across Calendars and Inboxes",
        "From Chatbot to Coworker: Building Useful Agents",
    ],
    "MLOps": [
        "Shipping LLM Apps: A CI/CD Playbook",
        "Observability for LLM Pipelines with Langfuse",
        "Cost Control for Production Inference",
        "Feature Stores Revisited for the LLM Era",
        "Monitoring Model Drift in the Wild",
        "Deployment Patterns for Agentic Systems",
        "Scaling Vector Search Infrastructure",
        "Incident Response for AI Systems",
    ],
    "Safety": [
        "Red Teaming Large Language Models",
        "Guardrails that Don't Get in the Way",
        "Data Verification Pipelines for Agent Outputs",
        "Privacy-Preserving Prompting Techniques",
        "Mitigating Prompt Injection in Tool-Using Agents",
        "Auditing AI Decisions for Compliance",
        "Model Cards and Transparency in Practice",
        "Building Trustworthy AI for Public Sector",
    ],
    "Applied AI": [
        "AI for Financial Services: A Case Study",
        "Conversational Interfaces for Healthcare",
        "AI-Assisted Customer Support at Scale",
        "Computer Vision on the Factory Floor",
        "Personalisation Without Creepiness",
        "AI in African Fintech: What's Working",
        "AI for Agricultural Yield Prediction",
        "Automating Legal Document Review with LLMs",
    ],
}

ORGANISATIONS = [
    "Nova Analytics",
    "Khanya AI Labs",
    "Dataforge",
    "Mzansi Machine Learning",
    "Horizon Cognition",
    "Thabo & Partners Consulting",
    "BrightPath AI",
    "Savanna Data Collective",
    "Quantum Ridge Technologies",
    "Ubuntu Intelligence",
    "Jacaranda Systems",
    "Silverline Robotics",
    "Nexus Cognitive Computing",
    "Kalahari Analytics",
    "Delta Forge AI",
    "Riverstone Technologies",
    "Vantage Point AI",
    "Clearwater Machine Intelligence",
    "Northstar Data Labs",
    "Ironwood AI Research",
]

JOB_TITLES = [
    "ML Engineer",
    "Head of AI",
    "Data Scientist",
    "Research Scientist",
    "Applied Scientist",
    "AI Product Manager",
    "Principal Engineer",
    "Founder & CEO",
    "Director of Engineering",
    "Developer Advocate",
    "Research Lead",
    "Solutions Architect",
    "VP of Engineering",
    "Staff Engineer",
]

COUNTRIES = [
    "South Africa",
    "Kenya",
    "Nigeria",
    "United Kingdom",
    "Germany",
    "United States",
    "Egypt",
    "Ghana",
    "Netherlands",
    "India",
]

FOCUS_AREAS = [
    "retrieval-augmented generation",
    "multi-agent systems",
    "MLOps and observability",
    "responsible AI",
    "natural language processing",
    "computer vision",
    "data verification pipelines",
    "LLM evaluation",
]

REGIONS = ["Africa", "Europe", "North America", "the Middle East", "Asia"]

PASSIONS = [
    "mentoring early-career engineers",
    "open-source tooling",
    "bridging research and production",
    "African tech ecosystems",
    "responsible AI adoption",
]

TICKET_TIERS = [
    {
        "tier": "Standard",
        "price_zar": "4,500",
        "includes": "Full conference access (Day 1-2), lunch, session recordings",
        "refund_policy": (
            "Full refund up to 14 days before the event; 50% refund up to 7 days before; "
            "no refund thereafter."
        ),
    },
    {
        "tier": "Student",
        "price_zar": "1,200",
        "includes": "Full conference access (Day 1-2), lunch",
        "refund_policy": "Full refund up to 14 days before the event; no refund thereafter.",
    },
    {
        "tier": "VIP",
        "price_zar": "8,900",
        "includes": "Full conference access, reserved seating, speaker dinner, session recordings",
        "refund_policy": (
            "Non-refundable; transferable to another attendee up to 48 hours before the event."
        ),
    },
    {
        "tier": "Virtual",
        "price_zar": "900",
        "includes": "Livestream access to Hall A and Hall B sessions, session recordings",
        "refund_policy": "Full refund up to 48 hours before the event.",
    },
]

KEY_POLICIES = [
    {
        "policy": "Catering cut-off",
        "rule": (
            "Lunch orders for dietary requirements must be submitted by 17:00 on "
            "11 November 2026. Walk-in catering is not guaranteed after this cut-off."
        ),
    },
    {
        "policy": "Recording consent",
        "rule": (
            "By entering session rooms, attendees consent to being recorded on video and "
            "audio for session playback. Attendees who do not wish to be recorded must sit "
            "in the designated no-recording section at the back of each room."
        ),
    },
    {
        "policy": "Badge rules",
        "rule": (
            "Badges must be worn and visible at all times inside the venue. Lost badges "
            "must be reported to the registration desk immediately and may incur a "
            "replacement fee of ZAR 150."
        ),
    },
    {
        "policy": "Late arrival",
        "rule": (
            "Attendees arriving more than 10 minutes after a session start time may be "
            "asked to wait for a suitable break before entering, to minimise disruption."
        ),
    },
]

# Fixed indices (into the 0-based, pre-sort session build order) used to plant
# the required edge cases deterministically.
I_FORCED_ACCESSIBLE = 0  # forced into the non-wheelchair-accessible room, Scheduled
I_COSPEAKER = 7
I_CANCELLED = 20
I_MOVED = 25
I_TOP_SPEAKER = (10, 11, 12)  # 3 sessions for the one speaker with the most
I_CONFUSABLE_A = (13, 14)  # SP-001 "Jon Mokoena"'s ordinary sessions
I_CONFUSABLE_B = (16, 17)  # SP-002 "John Mokoena"'s ordinary sessions

#: Speaker indices (0-based, into the speakers list) assigned to each
#: planted edge case. Every edge case gets its own speaker(s), disjoint from
#: every other edge case, so no single speaker carries more than one trap.
SPEAKER_CONFUSABLE_A = 0  # SP-001
SPEAKER_CONFUSABLE_B = 1  # SP-002
SPEAKER_FORCED_ACCESSIBLE = 2  # SP-003
SPEAKER_COSPEAKER_PRIMARY = 3  # SP-004
SPEAKER_MOVED = 5  # SP-006
SPEAKER_CANCELLED = 6  # SP-007
SPEAKER_TOP = 7  # SP-008
SPEAKER_COSPEAKER_EXTRA = 18  # SP-019

#: {edge_case_name: {speaker_ids involved}} - used to assert disjointness.
EDGE_CASE_SPEAKER_INDICES = {
    "confusable_pair": {SPEAKER_CONFUSABLE_A, SPEAKER_CONFUSABLE_B},
    "forced_accessible": {SPEAKER_FORCED_ACCESSIBLE},
    "cospeaker": {SPEAKER_COSPEAKER_PRIMARY, SPEAKER_COSPEAKER_EXTRA},
    "cancelled": {SPEAKER_CANCELLED},
    "moved": {SPEAKER_MOVED},
    "top_speaker": {SPEAKER_TOP},
}


def make_email(name: str, idx: int, used: set[str]) -> str:
    slug = re.sub(r"[^a-z.]", "", name.lower().replace(" ", "."))
    domain = "example.com" if idx % 2 == 0 else "example.org"
    email = f"{slug}@{domain}"
    n = 1
    while email in used:
        email = f"{slug}{n}@{domain}"
        n += 1
    used.add(email)
    return email


def make_phone(rng: random.Random, idx: int) -> str:
    fmt = ["e164", "local", "spaced", "dashed"][idx % 4]
    first = rng.choice("678")
    rest = "".join(rng.choice("0123456789") for _ in range(8))
    sub = first + rest
    if fmt == "e164":
        return f"+27{sub}"
    if fmt == "local":
        return f"0{sub}"
    if fmt == "spaced":
        return f"0{sub[0:2]} {sub[2:5]} {sub[5:9]}"
    return f"0{sub[0:2]}-{sub[2:5]}-{sub[5:9]}"


def make_bio(rng: random.Random, name: str, job_title: str, org: str) -> str:
    years = rng.randint(3, 18)
    focus = rng.choice(FOCUS_AREAS)
    region = rng.choice(REGIONS)
    passion = rng.choice(PASSIONS)
    first_name = name.split()[0]
    s1 = f"{name} is a {job_title} at {org} with {years} years of experience in applied AI."
    s2 = f"Their work focuses on {focus}, and they have presented at events across {region}."
    s3 = f"Outside of work, {first_name} is passionate about {passion}."
    return " ".join([s1, s2, s3]) if rng.random() < 0.5 else " ".join([s1, s2])


def make_linkedin(speaker_id: str, idx: int) -> str | None:
    if idx % 4 == 0:
        return None
    return f"atlas-demo-{speaker_id.lower()}"


def build_speakers(rng: random.Random, fake: Faker) -> list[dict]:
    special = [
        ("Jon Mokoena", "Nova Analytics", "Head of AI", "South Africa"),
        ("John Mokoena", "Ironwood AI Research", "Data Scientist", "South Africa"),
    ]
    special_names = {s[0] for s in special}

    names_rest: list[str] = []
    while len(names_rest) < 23:
        candidate = fake.name()
        if candidate not in special_names and candidate not in names_rest:
            names_rest.append(candidate)

    people = list(special) + [
        (name, rng.choice(ORGANISATIONS), rng.choice(JOB_TITLES), rng.choice(COUNTRIES))
        for name in names_rest
    ]

    used_emails: set[str] = set()
    speakers = []
    for idx, (name, org, job_title, country) in enumerate(people, start=1):
        speaker_id = f"SP-{idx:03d}"
        speakers.append(
            {
                "speaker_id": speaker_id,
                "full_name": name,
                "organisation": org,
                "job_title": job_title,
                "email": make_email(name, idx, used_emails),
                "phone": make_phone(rng, idx),
                "country": country,
                "bio": make_bio(rng, name, job_title, org),
                "linkedin_handle": make_linkedin(speaker_id, idx),
                "session_ids": [],
            }
        )
    return speakers


def build_sessions(rng: random.Random, speakers: list[dict]) -> list[dict]:
    combos = [
        {"day": day, "room_id": room["room_id"], "start_time": start, "end_time": end}
        for day in (1, 2)
        for room in ROOMS
        for start, end in SLOTS
    ]

    forced_combo = next(
        c
        for c in combos
        if c["room_id"] == "R-04" and c["day"] == 1 and c["start_time"] == SLOTS[0][0]
    )
    remaining = [c for c in combos if c is not forced_combo]
    rng.shuffle(remaining)
    chosen = [forced_combo, *remaining[:39]]

    track_title_pairs = [
        (TRACKS[i % 5], TRACK_TITLES[TRACKS[i % 5]][i // 5]) for i in range(40)
    ]

    rooms_by_id = {r["room_id"]: r for r in ROOMS}

    # --- speaker assignment plan ---
    # Every planted edge case gets its own reserved session slot(s) and its
    # own speaker(s) (see EDGE_CASE_SPEAKER_INDICES), so no speaker carries
    # more than one trap. All remaining ("ordinary") slots are filled by
    # round-robin over the remaining speakers, capped at 2 sessions each so
    # none of them can tie the designated top speaker's count of 3.
    reserved_primary: dict[int, int] = {
        I_FORCED_ACCESSIBLE: SPEAKER_FORCED_ACCESSIBLE,
        I_COSPEAKER: SPEAKER_COSPEAKER_PRIMARY,
        I_CANCELLED: SPEAKER_CANCELLED,
        I_MOVED: SPEAKER_MOVED,
        **dict.fromkeys(I_TOP_SPEAKER, SPEAKER_TOP),
        **dict.fromkeys(I_CONFUSABLE_A, SPEAKER_CONFUSABLE_A),
        **dict.fromkeys(I_CONFUSABLE_B, SPEAKER_CONFUSABLE_B),
    }
    extra_speaker_by_slot: dict[int, int] = {I_COSPEAKER: SPEAKER_COSPEAKER_EXTRA}

    reserved_speaker_indices = set().union(*EDGE_CASE_SPEAKER_INDICES.values())
    ordinary_speaker_indices = [i for i in range(25) if i not in reserved_speaker_indices]
    ordinary_slots = [i for i in range(40) if i not in reserved_primary]

    primary_by_slot: dict[int, int] = dict(reserved_primary)
    for pos, slot in enumerate(ordinary_slots):
        primary_by_slot[slot] = ordinary_speaker_indices[pos % len(ordinary_speaker_indices)]

    sessions = []
    for i in range(40):
        track, title = track_title_pairs[i]
        level = LEVELS[rng.randrange(3)]
        combo = chosen[i]
        room = rooms_by_id[combo["room_id"]]
        primary = speakers[primary_by_slot[i]]
        speaker_ids = [primary["speaker_id"]]
        role = None
        orig_room_id = None

        if i == I_FORCED_ACCESSIBLE:
            role = "forced_accessible"
        if i in extra_speaker_by_slot:
            extra = speakers[extra_speaker_by_slot[i]]
            speaker_ids.append(extra["speaker_id"])
            role = "cospeaker"

        status = "Scheduled"
        notes = ""
        if i == I_CANCELLED:
            status = "Cancelled"
            notes = "Cancelled due to speaker unavailability."
            role = "cancelled"
        elif i == I_MOVED:
            status = "Moved"
            orig_room_id = "R-03" if combo["room_id"] != "R-03" else "R-05"
            orig_room = rooms_by_id[orig_room_id]
            notes = (
                f"Moved from {orig_room['name']} ({orig_room['room_id']}) to "
                f"{room['name']} ({room['room_id']}) due to a venue scheduling conflict."
            )
            role = "moved"

        sessions.append(
            {
                "title": title,
                "track": track,
                "level": level,
                "day": combo["day"],
                "start_time": combo["start_time"],
                "end_time": combo["end_time"],
                "room_id": combo["room_id"],
                "speaker_ids": ";".join(speaker_ids),
                "capacity": room["capacity"],
                "status": status,
                "notes": notes,
                "_role": role,
                "_orig_room_id": orig_room_id,
            }
        )

    sessions.sort(key=lambda s: (s["day"], s["start_time"], s["room_id"]))
    for idx, s in enumerate(sessions, start=1):
        s["session_id"] = f"S-{idx:03d}"

    # Capacity: start from the room's capacity, but shave a small, growing
    # amount off each later use of the same room (first use of a room keeps
    # its full capacity). This keeps values realistic while guaranteeing a
    # single, unique global-maximum-capacity session: no other room's base
    # capacity comes close to R-01's 300, so R-01's first use is always the
    # unique max, even if it happens to be the Cancelled/Moved session.
    room_use_count: dict[str, int] = defaultdict(int)
    for s in sessions:
        room = rooms_by_id[s["room_id"]]
        use_index = room_use_count[s["room_id"]]
        room_use_count[s["room_id"]] += 1
        jitter = sum(rng.randint(1, 3) for _ in range(use_index))
        s["capacity"] = max(room["capacity"] - jitter, 10)

    return sessions


def assign_speaker_session_ids(speakers: list[dict], sessions: list[dict]) -> None:
    mapping: dict[str, list[str]] = defaultdict(list)
    for s in sessions:
        for sid in s["speaker_ids"].split(";"):
            mapping[sid].append(s["session_id"])
    for sp in speakers:
        sp["session_ids"] = sorted(mapping[sp["speaker_id"]])


def sanity_check(speakers: list[dict], sessions: list[dict]) -> None:
    speaker_ids = {sp["speaker_id"] for sp in speakers}
    session_ids = {s["session_id"] for s in sessions}
    room_ids = {r["room_id"] for r in ROOMS}

    for s in sessions:
        assert s["room_id"] in room_ids
        for sid in s["speaker_ids"].split(";"):
            assert sid in speaker_ids

    for sp in speakers:
        for sid in sp["session_ids"]:
            assert sid in session_ids
        assert sp["email"].endswith("@example.com") or sp["email"].endswith("@example.org")

    counts: dict[str, int] = defaultdict(int)
    going_counts: dict[str, int] = defaultdict(int)
    for s in sessions:
        for sid in s["speaker_ids"].split(";"):
            counts[sid] += 1
            if s["status"] != "Cancelled":
                going_counts[sid] += 1
    assert sum(1 for c in counts.values() if c >= 2) >= 2
    assert any(len(s["speaker_ids"].split(";")) >= 2 for s in sessions)
    assert any(s["status"] == "Cancelled" for s in sessions)
    assert any(s["status"] == "Moved" for s in sessions)

    non_accessible_rooms = {r["room_id"] for r in ROOMS if r["wheelchair_accessible"] == "No"}
    assert any(
        s["room_id"] in non_accessible_rooms and s["status"] == "Scheduled" for s in sessions
    )

    # Uniqueness: any "most/highest" gold question must have exactly one
    # correct answer among sessions that are going ahead.
    max_going_count = max(going_counts.values())
    assert sum(1 for c in going_counts.values() if c == max_going_count) == 1

    going_sessions = [s for s in sessions if s["status"] != "Cancelled"]
    max_capacity = max(s["capacity"] for s in going_sessions)
    assert sum(1 for s in going_sessions if s["capacity"] == max_capacity) == 1

    by_room_day: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for s in sessions:
        if s["status"] != "Cancelled":
            by_room_day[(s["room_id"], s["day"])].append((s["start_time"], s["end_time"]))
    for intervals in by_room_day.values():
        intervals.sort()
        for (_, end1), (start2, _) in zip(intervals, intervals[1:], strict=False):
            assert end1 <= start2

    # No speaker may carry more than one planted edge case: each edge case's
    # speaker(s) must be pairwise disjoint from every other edge case's.
    by_role = {s["_role"]: s for s in sessions if s.get("_role")}
    confusable_ids = {
        sp["speaker_id"] for sp in speakers if sp["full_name"] in ("Jon Mokoena", "John Mokoena")
    }
    top_speaker_id = next(sid for sid, c in going_counts.items() if c == max_going_count)
    edge_case_speakers = {
        "confusable_pair": confusable_ids,
        "forced_accessible": set(by_role["forced_accessible"]["speaker_ids"].split(";")),
        "cospeaker": set(by_role["cospeaker"]["speaker_ids"].split(";")),
        "cancelled": set(by_role["cancelled"]["speaker_ids"].split(";")),
        "moved": set(by_role["moved"]["speaker_ids"].split(";")),
        "top_speaker": {top_speaker_id},
    }
    names = list(edge_case_speakers)
    for a, b in combinations(names, 2):
        overlap = edge_case_speakers[a] & edge_case_speakers[b]
        assert not overlap, f"edge cases {a!r} and {b!r} share speaker(s) {overlap}"


def write_sessions_csv(sessions: list[dict], path: Path) -> None:
    fieldnames = [
        "session_id",
        "title",
        "track",
        "level",
        "day",
        "start_time",
        "end_time",
        "room_id",
        "speaker_ids",
        "capacity",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for s in sessions:
            writer.writerow(s)


def write_speakers_json(speakers: list[dict], path: Path) -> None:
    clean = [{k: v for k, v in sp.items() if not k.startswith("_")} for sp in speakers]
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def write_venue_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    cell_style = styles["BodyText"]
    cell_style.fontSize = 8
    cell_style.leading = 10

    header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )

    def p(text: str) -> Paragraph:
        return Paragraph(text, cell_style)

    def h(text: str) -> Paragraph:
        return Paragraph(text, header_style)

    story = [
        Paragraph(f"{CONFERENCE_NAME} &mdash; Venue Guide", styles["Title"]),
        Spacer(1, 6 * mm),
        Paragraph(
            f"The {CONFERENCE_NAME} is a two-day conference held on 18-19 November 2026 in "
            f"{VENUE_CITY}, South Africa. This guide covers the venue's rooms, ticket tiers "
            "and key attendee policies. This is a fictional event created for a retrieval "
            "evaluation exercise; all people and organisations referenced elsewhere in this "
            "knowledge base are fictional.",
            styles["BodyText"],
        ),
        PageBreak(),
        Paragraph("Rooms", styles["Heading2"]),
    ]

    room_header = [
        h("Room ID"),
        h("Name"),
        h("Floor"),
        h("Capacity"),
        h("Wheelchair Accessible"),
        h("AV Equipment"),
        h("Hearing Loop"),
    ]
    room_rows = [room_header] + [
        [
            r["room_id"],
            r["name"],
            str(r["floor"]),
            str(r["capacity"]),
            r["wheelchair_accessible"],
            p(r["av_equipment"]),
            r["hearing_loop"],
        ]
        for r in ROOMS
    ]
    room_col_widths = [18 * mm, 24 * mm, 12 * mm, 18 * mm, 36 * mm, 46 * mm, 24 * mm]
    room_table = Table(room_rows, colWidths=room_col_widths, repeatRows=1)
    room_table.setStyle(_table_style())
    story.append(room_table)
    story.append(PageBreak())

    story.append(Paragraph("Ticket Tiers", styles["Heading2"]))
    tier_header = [h("Tier"), h("Price (ZAR)"), h("Includes"), h("Refund Policy")]
    tier_rows = [tier_header] + [
        [t["tier"], t["price_zar"], p(t["includes"]), p(t["refund_policy"])] for t in TICKET_TIERS
    ]
    tier_table = Table(tier_rows, colWidths=[20 * mm, 20 * mm, 60 * mm, 78 * mm], repeatRows=1)
    tier_table.setStyle(_table_style())
    story.append(tier_table)
    story.append(PageBreak())

    story.append(Paragraph("Key Policies", styles["Heading2"]))
    policy_header = [h("Policy"), h("Rule")]
    policy_rows = [policy_header] + [[pol["policy"], p(pol["rule"])] for pol in KEY_POLICIES]
    policy_table = Table(policy_rows, colWidths=[30 * mm, 148 * mm], repeatRows=1)
    policy_table.setStyle(_table_style())
    story.append(policy_table)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=f"{CONFERENCE_NAME} Venue Guide",
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )
    doc.build(story, canvasmaker=partial(Canvas, invariant=1))


#: Paraphrased session-topic questions for the "semantic" gold items. Each
#: question deliberately avoids every distinctive word in the session title
#: it targets, and never mentions an ID, so it can only be answered by
#: understanding the topic rather than matching keywords.
SEMANTIC_QUESTIONS = [
    (
        "Privacy-Preserving Prompting Techniques",
        "Which talk covers how to keep user data confidential when crafting "
        "inputs for a language model?",
    ),
    (
        "Reducing Hallucinations with Grounded Retrieval",
        "Which session discusses making a language model's made-up answers "
        "less common by anchoring its output in real source material?",
    ),
    (
        "Mitigating Prompt Injection in Tool-Using Agents",
        "Which talk addresses defending AI systems that call external tools "
        "from attackers sneaking malicious instructions into their inputs?",
    ),
    (
        "Cost Control for Production Inference",
        "Which session is about keeping spending in check when running "
        "language models live for users?",
    ),
    (
        "AI for Agricultural Yield Prediction",
        "Which talk explores forecasting how much a farm's crops will produce?",
    ),
]

#: Same idea as SEMANTIC_QUESTIONS, but targeting two different session
#: titles entirely, reserved for the held-out set (see generate_heldout).
#: Reset after kb_version 1.0.2 (see CHANGELOG.md): the 1.0.1 held-out set
#: was accidentally evaluated once during development and was retired
#: without informing any tuning decision. These two titles were not used by
#: either SEMANTIC_QUESTIONS or the retired 1.0.1 held-out set.
HELDOUT_SEMANTIC_QUESTIONS = [
    (
        "Vector Databases: A Practical Comparison",
        "Which talk evaluates several storage systems for embeddings and "
        "weighs how they perform in real deployments?",
    ),
    (
        "Monitoring Model Drift in the Wild",
        "Which session covers watching for a deployed AI system's "
        "behaviour gradually changing after release?",
    ),
]


def generate_gold(
    sessions: list[dict], speakers: list[dict], sessions_csv_path: Path
) -> list[dict]:
    df = pd.read_csv(sessions_csv_path, keep_default_na=False)
    # Counting convention: aggregations exclude Cancelled sessions unless the
    # question is explicitly about cancellations (Moved sessions still count,
    # since they are going ahead, just relocated).
    df_going = df[df["status"] != "Cancelled"]

    by_role = {s["_role"]: s for s in sessions if s.get("_role")}
    forced = by_role["forced_accessible"]
    cospeaker_s = by_role["cospeaker"]
    cancelled_s = by_role["cancelled"]
    moved_s = by_role["moved"]

    rooms_by_id = {r["room_id"]: r for r in ROOMS}
    speakers_by_id = {sp["speaker_id"]: sp for sp in speakers}
    title_to_session = {s["title"]: s for s in sessions}

    items: list[dict] = []

    def add(
        question: str,
        answer: str,
        value: object,
        answer_type: str,
        refs: list[str],
        tolerance: float | None = None,
    ) -> None:
        item = {
            "id": f"G-{len(items) + 1:03d}",
            "question": question,
            "expected_answer": answer,
            "expected_value": value,
            "answer_type": answer_type,
            "source_refs": refs,
        }
        if tolerance is not None:
            item["tolerance"] = tolerance
        items.append(item)

    # --- lookup (6), including >=1 cancelled and >=1 moved ---
    s1 = sessions[0]
    add(
        f"What is the title of session {s1['session_id']}?",
        s1["title"],
        s1["title"],
        "lookup",
        [f"sessions.csv#{s1['session_id']}"],
    )

    add(
        f"Which room is session {forced['session_id']} held in?",
        f"{rooms_by_id[forced['room_id']]['name']} ({forced['room_id']})",
        forced["room_id"],
        "lookup",
        [f"sessions.csv#{forced['session_id']}"],
    )

    sp6 = speakers[8]  # an ordinary speaker, not involved in any planted edge case
    add(
        f"What is the email address of speaker {sp6['speaker_id']}?",
        sp6["email"],
        sp6["email"],
        "lookup",
        [f"speakers.json#{sp6['speaker_id']}"],
    )

    add(
        f"What is the status of session {cancelled_s['session_id']}?",
        cancelled_s["status"],
        cancelled_s["status"],
        "lookup",
        [f"sessions.csv#{cancelled_s['session_id']}"],
    )

    orig_room = rooms_by_id[moved_s["_orig_room_id"]]
    add(
        f"Before it moved, which room was session {moved_s['session_id']} originally scheduled in?",
        f"{orig_room['name']} ({orig_room['room_id']})",
        orig_room["room_id"],
        "lookup",
        [f"sessions.csv#{moved_s['session_id']}"],
    )

    sp2 = speakers_by_id["SP-002"]
    add(
        f"What organisation is speaker {sp2['speaker_id']} affiliated with?",
        sp2["organisation"],
        sp2["organisation"],
        "lookup",
        [f"speakers.json#{sp2['speaker_id']}"],
    )

    # --- aggregation (6), computed via the CSV with pandas; Cancelled
    # sessions are excluded except where the question is explicitly about
    # cancellations ---
    day1_count = int((df_going["day"] == 1).sum())
    add(
        "How many sessions are going ahead on Day 1 (18 November 2026)?",
        str(day1_count),
        day1_count,
        "aggregation",
        ["sessions.csv"],
    )

    safety_count = int((df_going["track"] == "Safety").sum())
    add(
        "How many sessions in the Safety track are going ahead?",
        str(safety_count),
        safety_count,
        "aggregation",
        ["sessions.csv"],
    )

    avg_capacity = round(float(df_going["capacity"].mean()), 1)
    add(
        "What is the average capacity of sessions that are going ahead, rounded to 1 "
        "decimal place?",
        str(avg_capacity),
        avg_capacity,
        "aggregation",
        ["sessions.csv"],
        tolerance=0.05,
    )

    top = df_going.sort_values(["capacity", "session_id"], ascending=[False, True]).iloc[0]
    add(
        "Which session (by ID) has the highest capacity among sessions that are going ahead?",
        f"{top['session_id']} ({top['title']}) with a capacity of {int(top['capacity'])}.",
        top["session_id"],
        "aggregation",
        [f"sessions.csv#{top['session_id']}"],
    )

    cancelled_count = int((df["status"] == "Cancelled").sum())
    add(
        "How many sessions have a status of Cancelled?",
        str(cancelled_count),
        cancelled_count,
        "aggregation",
        ["sessions.csv"],
    )

    sp1_count = int(df_going["speaker_ids"].apply(lambda v: "SP-001" in v.split(";")).sum())
    add(
        "How many sessions that are going ahead is speaker SP-001 presenting (as a speaker "
        "or co-speaker)?",
        str(sp1_count),
        sp1_count,
        "aggregation",
        ["sessions.csv", "speakers.json#SP-001"],
    )

    # --- cross_doc (5) ---
    clean_sessions = [s for s in sessions if not s.get("_role")]
    cd1 = clean_sessions[0]
    speaker_cd1 = speakers_by_id[cd1["speaker_ids"].split(";")[0]]
    add(
        f"Which organisation does the speaker presenting session {cd1['session_id']} work for?",
        speaker_cd1["organisation"],
        speaker_cd1["organisation"],
        "cross_doc",
        [f"sessions.csv#{cd1['session_id']}", f"speakers.json#{speaker_cd1['speaker_id']}"],
    )

    forced_room = rooms_by_id[forced["room_id"]]
    accessible = forced_room["wheelchair_accessible"]
    add(
        f"Is the room used for session {forced['session_id']} wheelchair accessible?",
        f"{accessible} ({forced_room['name']} is "
        f"{'not ' if accessible == 'No' else ''}wheelchair accessible).",
        accessible,
        "cross_doc",
        [f"sessions.csv#{forced['session_id']}", "venue_guide.pdf#rooms"],
    )

    co_ids = cospeaker_s["speaker_ids"].split(";")
    co_speakers = [speakers_by_id[sid] for sid in co_ids]
    names_orgs = "; ".join(f"{sp['full_name']} ({sp['organisation']})" for sp in co_speakers)
    add(
        f"Who are the co-speakers for session {cospeaker_s['session_id']}, and what are "
        "their organisations?",
        names_orgs,
        ";".join(sorted(co_ids)),
        "cross_doc",
        [f"sessions.csv#{cospeaker_s['session_id']}"] + [f"speakers.json#{sid}" for sid in co_ids],
    )

    # Counting convention applies here too: a speaker's cancelled sessions
    # don't count toward "sessions given".
    counts: dict[str, int] = defaultdict(int)
    for s in sessions:
        if s["status"] == "Cancelled":
            continue
        for sid in s["speaker_ids"].split(";"):
            counts[sid] += 1
    max_count = max(counts.values())
    top_speaker_id = sorted(sid for sid, c in counts.items() if c == max_count)[0]
    top_speaker = speakers_by_id[top_speaker_id]
    add(
        "Which speaker (by ID) has given the most sessions that are going ahead, and what "
        "is their job title?",
        f"{top_speaker_id} ({top_speaker['full_name']}, {top_speaker['job_title']}) with "
        f"{max_count} sessions.",
        top_speaker_id,
        "aggregation",
        [f"speakers.json#{top_speaker_id}", "sessions.csv"],
    )

    cancelled_room = rooms_by_id[cancelled_s["room_id"]]
    add(
        f"Does the room hosting the cancelled session {cancelled_s['session_id']} have a "
        "hearing loop?",
        f"{cancelled_room['hearing_loop']} ({cancelled_room['name']}).",
        cancelled_room["hearing_loop"],
        "cross_doc",
        [f"sessions.csv#{cancelled_s['session_id']}", "venue_guide.pdf#rooms"],
    )

    # --- policy (4) ---
    policy_map = {p["policy"]: p["rule"] for p in KEY_POLICIES}
    add(
        "What is the venue's catering cut-off policy?",
        policy_map["Catering cut-off"],
        policy_map["Catering cut-off"],
        "policy",
        ["venue_guide.pdf#policies"],
    )
    add(
        f"What is the recording consent policy at the {CONFERENCE_NAME}?",
        policy_map["Recording consent"],
        policy_map["Recording consent"],
        "policy",
        ["venue_guide.pdf#policies"],
    )
    add(
        "What happens if an attendee loses their conference badge?",
        policy_map["Badge rules"],
        policy_map["Badge rules"],
        "policy",
        ["venue_guide.pdf#policies"],
    )
    add(
        "What is the policy for attendees who arrive late to a session?",
        policy_map["Late arrival"],
        policy_map["Late arrival"],
        "policy",
        ["venue_guide.pdf#policies"],
    )

    # --- negative (4): nothing in the KB backs these, so expected_value is
    # the literal sentinel NOT_IN_KB ---
    add(
        "What is the Wi-Fi password at the venue?",
        "Not in the knowledge base.",
        "NOT_IN_KB",
        "negative",
        [],
    )
    add(
        f"Is there a Day 3 of the {CONFERENCE_NAME}?",
        "Not in the knowledge base.",
        "NOT_IN_KB",
        "negative",
        [],
    )
    add(
        f"What hotel discount code is available for attendees of the {CONFERENCE_NAME}?",
        "Not in the knowledge base.",
        "NOT_IN_KB",
        "negative",
        [],
    )
    add(
        "Who is the keynote speaker for the closing ceremony?",
        "Not in the knowledge base.",
        "NOT_IN_KB",
        "negative",
        [],
    )

    # --- semantic (5): paraphrases with no shared key terms and no IDs ---
    for title, question in SEMANTIC_QUESTIONS:
        session = title_to_session[title]
        add(
            question,
            f"{session['session_id']} ({session['title']})",
            session["session_id"],
            "semantic",
            [f"sessions.csv#{session['session_id']}"],
        )

    return items


#: The kb_version 1.0.1 held-out set's exact source_refs, retired after it
#: was accidentally evaluated once during development (see CHANGELOG.md).
#: Excluded here too, so the reset set shares no entity with either the
#: gold set or the retired set.
RETIRED_HELDOUT_SESSION_IDS = {"S-002", "S-003", "S-005", "S-006", "S-007", "S-033"}
RETIRED_HELDOUT_SPEAKER_IDS = {"SP-003", "SP-018"}


def generate_heldout(
    sessions: list[dict],
    speakers: list[dict],
    sessions_csv_path: Path,
    used_refs: set[str],
) -> list[dict]:
    """~15 fresh items held out from evals/kb_gold.jsonl: computed from the
    same frozen data, but none of them paraphrase an existing gold item, and
    none reuse an entity from the retired kb_version 1.0.1 held-out set
    either (see RETIRED_HELDOUT_SESSION_IDS/SPEAKER_IDS). Evaluated once, at
    submission time (see README "Evaluation")."""
    df = pd.read_csv(sessions_csv_path, keep_default_na=False)
    df_going = df[df["status"] != "Cancelled"]

    speakers_by_id = {sp["speaker_id"]: sp for sp in speakers}
    rooms_by_id = {r["room_id"]: r for r in ROOMS}
    title_to_session = {s["title"]: s for s in sessions}
    policy_map = {p["policy"]: p["rule"] for p in KEY_POLICIES}

    used_session_ids = {
        ref.split("#", 1)[1] for ref in used_refs if ref.startswith("sessions.csv#")
    } | RETIRED_HELDOUT_SESSION_IDS
    used_speaker_ids = {
        ref.split("#", 1)[1] for ref in used_refs if ref.startswith("speakers.json#")
    } | RETIRED_HELDOUT_SPEAKER_IDS
    unused_sessions = [s for s in sessions if s["session_id"] not in used_session_ids]
    unused_speakers = [sp for sp in speakers if sp["speaker_id"] not in used_speaker_ids]

    items: list[dict] = []

    def add(question: str, answer: str, value: object, answer_type: str, refs: list[str]) -> None:
        items.append(
            {
                "id": f"H-{len(items) + 1:03d}",
                "question": question,
                "expected_answer": answer,
                "expected_value": value,
                "answer_type": answer_type,
                "source_refs": refs,
            }
        )

    # --- lookup (3) ---
    s_title = unused_sessions[0]
    add(
        f"What is the title of session {s_title['session_id']}?",
        s_title["title"],
        s_title["title"],
        "lookup",
        [f"sessions.csv#{s_title['session_id']}"],
    )

    s_room = unused_sessions[1]
    room = rooms_by_id[s_room["room_id"]]
    add(
        f"Which room is session {s_room['session_id']} held in?",
        f"{room['name']} ({s_room['room_id']})",
        s_room["room_id"],
        "lookup",
        [f"sessions.csv#{s_room['session_id']}"],
    )

    sp_org = unused_speakers[0]
    add(
        f"What organisation is speaker {sp_org['speaker_id']} affiliated with?",
        sp_org["organisation"],
        sp_org["organisation"],
        "lookup",
        [f"speakers.json#{sp_org['speaker_id']}"],
    )

    # --- aggregation (3): fresh dimensions (not Day/Safety-track/top-speaker,
    # and not Day2/Agents-track/Beginner-level from the retired set) ---
    mlops_count = int((df_going["track"] == "MLOps").sum())
    add(
        "How many sessions in the MLOps track are going ahead?",
        str(mlops_count),
        mlops_count,
        "aggregation",
        ["sessions.csv"],
    )

    advanced_count = int((df_going["level"] == "Advanced").sum())
    add(
        "How many Advanced-level sessions are going ahead?",
        str(advanced_count),
        advanced_count,
        "aggregation",
        ["sessions.csv"],
    )

    rag_avg_capacity = round(float(df_going.loc[df_going["track"] == "RAG", "capacity"].mean()), 1)
    add(
        "What is the average capacity of RAG track sessions that are going ahead, "
        "rounded to 1 decimal place?",
        str(rag_avg_capacity),
        rag_avg_capacity,
        "aggregation",
        ["sessions.csv"],
    )

    # --- cross_doc (3) ---
    s_cd1 = unused_sessions[2]
    speaker_cd1 = speakers_by_id[s_cd1["speaker_ids"].split(";")[0]]
    add(
        f"Which organisation does the speaker presenting session {s_cd1['session_id']} work for?",
        speaker_cd1["organisation"],
        speaker_cd1["organisation"],
        "cross_doc",
        [f"sessions.csv#{s_cd1['session_id']}", f"speakers.json#{speaker_cd1['speaker_id']}"],
    )

    s_cd2 = unused_sessions[3]
    room_cd2 = rooms_by_id[s_cd2["room_id"]]
    add(
        f"Is the room used for session {s_cd2['session_id']} wheelchair accessible?",
        f"{room_cd2['wheelchair_accessible']} ({room_cd2['name']}).",
        room_cd2["wheelchair_accessible"],
        "cross_doc",
        [f"sessions.csv#{s_cd2['session_id']}", "venue_guide.pdf#rooms"],
    )

    s_cd3 = unused_sessions[4]
    room_cd3 = rooms_by_id[s_cd3["room_id"]]
    add(
        f"Does the room hosting session {s_cd3['session_id']} have a hearing loop?",
        f"{room_cd3['hearing_loop']} ({room_cd3['name']}).",
        room_cd3["hearing_loop"],
        "cross_doc",
        [f"sessions.csv#{s_cd3['session_id']}", "venue_guide.pdf#rooms"],
    )

    # --- policy (2): fresh extracted sub-facts (not catering-time/badge-fee
    # from the retired set), not a reworded "what is policy X" ---
    late_rule = policy_map["Late arrival"]
    minutes = re.search(r"more than (\d+) minutes", late_rule)
    assert minutes, "late arrival policy text no longer matches the expected format"
    add(
        "How many minutes after a session's start time might a late attendee be "
        "asked to wait for a break before entering?",
        minutes.group(1),
        int(minutes.group(1)),
        "policy",
        ["venue_guide.pdf#policies"],
    )

    consent_rule = policy_map["Recording consent"]
    seating = re.search(r"must sit in (.+?)\.", consent_rule)
    assert seating, "recording consent policy text no longer matches the expected format"
    add(
        "Where must an attendee who doesn't want to be recorded sit?",
        seating.group(1),
        seating.group(1),
        "policy",
        ["venue_guide.pdf#policies"],
    )

    # --- semantic (2): two session titles not covered by SEMANTIC_QUESTIONS
    # or the retired held-out set ---
    for title, question in HELDOUT_SEMANTIC_QUESTIONS:
        session = title_to_session[title]
        add(
            question,
            f"{session['session_id']} ({session['title']})",
            session["session_id"],
            "semantic",
            [f"sessions.csv#{session['session_id']}"],
        )

    # --- negative (2) ---
    add(
        "Is there a cloakroom for luggage storage at the venue?",
        "Not in the knowledge base.",
        "NOT_IN_KB",
        "negative",
        [],
    )
    add(
        f"What is the {CONFERENCE_NAME}'s carbon offset program?",
        "Not in the knowledge base.",
        "NOT_IN_KB",
        "negative",
        [],
    )

    return items


def write_gold_jsonl(items: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True))
            f.write("\n")


def write_manifest(path: Path, kb_version: str, seed: int, files: dict[str, Path]) -> None:
    manifest = {
        "kb_version": kb_version,
        "generator_seed": seed,
        "files": {
            name: {"sha256": hashlib.sha256(file_path.read_bytes()).hexdigest()}
            for name, file_path in files.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def make_rng(seed: int, table: str) -> random.Random:
    return random.Random(f"{seed}:{table}")


def make_faker(seed: int, table: str) -> Faker:
    fake = Faker()
    fake.seed_instance(f"{seed}:{table}")
    return fake


def generate(kb_dir: Path, evals_dir: Path, seed: int = SEED) -> dict:
    kb_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    # Independent RNG (and Faker) per table: changing what one table
    # generates can never shift what any other table draws.
    rng_speakers = make_rng(seed, "speakers")
    rng_sessions = make_rng(seed, "sessions")
    fake_speakers = make_faker(seed, "speakers")

    speakers = build_speakers(rng_speakers, fake_speakers)
    sessions = build_sessions(rng_sessions, speakers)
    assign_speaker_session_ids(speakers, sessions)
    sanity_check(speakers, sessions)

    sessions_csv_path = kb_dir / "sessions.csv"
    speakers_json_path = kb_dir / "speakers.json"
    venue_pdf_path = kb_dir / "venue_guide.pdf"

    write_sessions_csv(sessions, sessions_csv_path)
    write_speakers_json(speakers, speakers_json_path)
    write_venue_pdf(venue_pdf_path)

    gold_items = generate_gold(sessions, speakers, sessions_csv_path)
    gold_jsonl_path = evals_dir / "kb_gold.jsonl"
    write_gold_jsonl(gold_items, gold_jsonl_path)

    used_refs = {ref for item in gold_items for ref in item["source_refs"]}
    heldout_items = generate_heldout(sessions, speakers, sessions_csv_path, used_refs)
    heldout_jsonl_path = evals_dir / "kb_heldout.jsonl"
    write_gold_jsonl(heldout_items, heldout_jsonl_path)

    write_manifest(
        kb_dir / "manifest.json",
        kb_version=KB_VERSION,
        seed=seed,
        files={
            "sessions.csv": sessions_csv_path,
            "speakers.json": speakers_json_path,
            "venue_guide.pdf": venue_pdf_path,
            "kb_gold.jsonl": gold_jsonl_path,
            "kb_heldout.jsonl": heldout_jsonl_path,
        },
    )

    return {
        "sessions": len(sessions),
        "speakers": len(speakers),
        "gold_items": len(gold_items),
        "heldout_items": len(heldout_items),
    }


def main() -> None:
    summary = generate(KB_DIR, EVALS_DIR, SEED)
    print(
        f"Generated {summary['sessions']} sessions, {summary['speakers']} speakers, "
        f"{summary['gold_items']} gold items, {summary['heldout_items']} held-out items."
    )


if __name__ == "__main__":
    main()
