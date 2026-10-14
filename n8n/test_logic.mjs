// Offline test harness for the normalise / verify+dedupe / score logic used
// by n8n/atlas_attendee_agent.json's Code nodes.
//
// This does NOT call the n8n workflow or the Anthropic API. It re-implements
// the same normalise/verify/dedupe/score functions found in n8n/code/*.js
// (07_normalise.js, 08_tag_directory.js, 09_verify_and_dedupe.js,
// 11_score_vs_answer_key.js) as plain functions, and replaces the LLM
// extraction step with a small regex-based mock extractor that reads the
// same 9 RSVP emails and 2 calendar events described in
// data/synthetic/part2_seed.md (copied here verbatim - see the comment
// above EMAILS below if the two ever drift).
//
// It also unit-tests getEmailText() (duplicated from
// n8n/code/01_build_extraction_prompt.js) against a synthetic raw Gmail API
// message fixture - a nested multipart/mixed -> multipart/alternative
// structure with base64url-encoded text/plain and text/html parts - since
// that function was the fix for a live-run bug where the real Gmail
// message shape wasn't being decoded at all (see the PR/commit history for
// the original diagnosis).
//
// No network calls: speakers.json and the answer key are read from the
// local filesystem, not fetched from GitHub.
//
// Run with: node n8n/test_logic.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

// ---------------------------------------------------------------------
// getEmailText (mirrors n8n/code/01_build_extraction_prompt.js and
// n8n/code/03_build_retry_prompt.js)
// ---------------------------------------------------------------------

function encodeBase64Url(text) {
  return Buffer.from(text, "utf8").toString("base64").replace(/\+/g, "-").replace(/\//g, "_");
}

function decodeBase64Url(data) {
  if (!data) return "";
  const normalized = data.replace(/-/g, "+").replace(/_/g, "/");
  return Buffer.from(normalized, "base64").toString("utf8");
}

function stripHtml(html) {
  return html
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .trim();
}

function findPartData(part, mimeType) {
  if (!part) return null;
  if (part.mimeType === mimeType && part.body && part.body.data) {
    return part.body.data;
  }
  if (Array.isArray(part.parts)) {
    for (const sub of part.parts) {
      const found = findPartData(sub, mimeType);
      if (found) return found;
    }
  }
  return null;
}

function getEmailText(item) {
  if (item.text && String(item.text).trim()) {
    return { text: String(item.text).trim(), source: "text" };
  }
  if (item.textPlain && String(item.textPlain).trim()) {
    return { text: String(item.textPlain).trim(), source: "textPlain" };
  }

  const payload = item.payload;

  const plainPartData = payload ? findPartData(payload, "text/plain") : null;
  if (plainPartData) {
    const decoded = decodeBase64Url(plainPartData).trim();
    if (decoded) return { text: decoded, source: "payload.parts(text/plain)" };
  }

  if (payload && payload.body && payload.body.data) {
    const decoded = decodeBase64Url(payload.body.data).trim();
    if (decoded) return { text: decoded, source: "payload.body.data" };
  }

  const htmlPartData = payload ? findPartData(payload, "text/html") : null;
  if (htmlPartData) {
    const decoded = stripHtml(decodeBase64Url(htmlPartData));
    if (decoded) return { text: decoded, source: "payload.parts(text/html)" };
  }

  if (item.snippet && String(item.snippet).trim()) {
    return { text: String(item.snippet).trim(), source: "snippet" };
  }

  return { text: "", source: "none" };
}

// A synthetic raw Gmail API message: multipart/mixed (as if it had an
// attachment slot) wrapping multipart/alternative with text/plain and
// text/html parts, both base64url-encoded - the shape the live run showed
// Gmail - Get Many actually returns with Simplify OFF.
const RAW_GMAIL_FIXTURE_PLAIN_TEXT =
  "Hi,\n\nConfirming attendance.\n\nFixture Person\nFixture Org\nfixture.person@example.com\n082 000 0000";

const RAW_GMAIL_FIXTURE = {
  id: "fixture-message-id-1",
  threadId: "fixture-thread-id-1",
  labelIds: ["INBOX"],
  snippet: "Hi, Confirming attendance. Fixture Person Fixture Org...",
  payload: {
    mimeType: "multipart/mixed",
    headers: [
      { name: "Subject", value: "RSVP - Fixture Test" },
      { name: "From", value: "someone@gmail.com" },
    ],
    parts: [
      {
        mimeType: "multipart/alternative",
        parts: [
          {
            mimeType: "text/plain",
            body: { data: encodeBase64Url(RAW_GMAIL_FIXTURE_PLAIN_TEXT) },
          },
          {
            mimeType: "text/html",
            body: {
              data: encodeBase64Url(
                "<div dir=\"ltr\">Hi,<br><br>Confirming attendance.<br><br>Fixture Person<br>Fixture Org<br>fixture.person@example.com<br>082 000 0000</div>"
              ),
            },
          },
        ],
      },
    ],
  },
};

// A second fixture with no decodable body at all, to prove the snippet
// fallback at the end of the chain actually fires instead of returning "".
const RAW_GMAIL_FIXTURE_SNIPPET_ONLY = {
  id: "fixture-message-id-2",
  threadId: "fixture-thread-id-2",
  labelIds: ["INBOX"],
  snippet: "Snippet-only fallback text",
  payload: {
    mimeType: "text/plain",
    headers: [{ name: "Subject", value: "No body" }],
    body: {},
  },
};

// ---------------------------------------------------------------------
// Seed data (verbatim from data/synthetic/part2_seed.md)
// ---------------------------------------------------------------------

const EMAILS = [
  {
    subject: "RSVP - Atlas AI Summit",
    body: `Hi,

Confirming I'll be attending the Atlas AI Summit on 18-19 November.
Looking forward to it!

Best,
Troy Harris
Quantum Ridge Technologies
troy.harris@example.org
062 629 5209`,
  },
  {
    subject: "re: Atlas AI Summit attendance",
    body: `Hey team

Yep I'm in for the summit, both days. Here are my details in case you
need them for badges:

Andrew Kane
Jacaranda Systems
andrew.kane@example.org
tel: 087 159 2445

See you there`,
  },
  {
    subject: "RSVP: Atlas AI Summit",
    body: `Hello,

I'd like to confirm my attendance at the Atlas AI Summit in November.

Thanks,
Latoya Wilson
Silver Line Robotics
latoya.wilson@example.com`,
  },
  {
    subject: "Atlas Summit - confirming attendance",
    body: `Hi there,

Just confirming I'll be at the Atlas AI Summit this November. You can
reach me on 073-359-1551 if anything changes.

Regards,
Joseph Torres
Horizon Cognition`,
  },
  {
    subject: "RSVP - Atlas AI Summit 2026",
    body: `Hi,

I'll be attending the Atlas AI Summit on both days. My details:

John Mokoena
Nova Analytics
john.mokoena@example.org
083 400 0938

Thanks,
John`,
  },
  {
    subject: "Re: Atlas AI Summit - RSVP",
    body: `Hi,

Confirming attendance for the Atlas AI Summit. My contact info:

Leslie Frey
Khanya AI Labs
lesliefrey@example.com
084-595-7566

Cheers`,
  },
  {
    subject: "Atlas AI Summit - RSVP",
    body: `Hi there,

I saw the summit announcement and would like to attend. Not a speaker,
just keen to attend as a delegate.

Priya Naidoo
DataForge Analytics
priya.naidoo@example.com
082 310 4456`,
  },
  {
    subject: "RSVP - Atlas AI Summit",
    body: `Hi,

Confirming I'll attend the Atlas AI Summit on 18-19 November.

Paige Williams
Nova Analytics
paige.williams@example.com
+27643819886`,
  },
  {
    subject: "Atlas AI Summit - RSVP (sorry, resending)",
    body: `Hi again,

Sorry, I don't think my earlier RSVP went through properly, resending.

Paige Willams
Nova Analytics
paige.williams@example.com
+27643819886

Thanks!`,
  },
];

const CALENDAR_EVENTS = [
  {
    summary: "Atlas AI Summit - Welcome Reception",
    description: `Welcome reception for all Atlas AI Summit attendees and speakers.

Confirmed attendees include:
- Troy Harris, Quantum Ridge Technologies
- Latoya Wilson, Silverline Robotics
- Paige Williams, Nova Analytics`,
  },
  {
    summary: "Atlas AI Summit - Closing Networking Lunch",
    description: `Closing lunch, open to all attendees.

Confirmed attendees include:
- Andrew Kane, Jacaranda Systems
- Leslie Frey, Khanya AI Labs
- Joseph Torres, Horizon Cognitions`,
  },
];

// ---------------------------------------------------------------------
// Mock extraction (regex-based stand-in for the Anthropic call)
// ---------------------------------------------------------------------

const EMAIL_RE = /[^\s@]+@[^\s@]+\.[^\s@]+/;
const PHONE_RE = /(\+?\d[\d\s\-()]{6,}\d)/;
const SIGNOFF_RE =
  /^(hi|hi there|hi again|hello|hey|hey team|thanks|thank you|regards|best|cheers|sincerely|see you there)[,!]?$/i;

function mockExtract(subject, body) {
  const lines = body
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  // Contact fields: body-wide regex scan, independent of line position.
  let email = null;
  let phone = null;
  for (const line of lines) {
    if (!email) {
      const m = line.match(EMAIL_RE);
      if (m) email = m[0];
    }
  }
  for (const line of lines) {
    if (!phone && !EMAIL_RE.test(line)) {
      const m = line.match(PHONE_RE);
      if (m) phone = m[0].trim();
    }
  }

  // Name/organisation: bottom-up scan over the trailing signature block,
  // skipping contact-info lines and short sign-offs (and a short line
  // immediately after a sign-off, e.g. "Thanks,\nJohn", but only when it's
  // the very last line of the body - otherwise a line like "Thanks,\nLatoya
  // Wilson\nOrg\nemail" would wrongly discard the real name).
  const skip = new Array(lines.length).fill(false);
  lines.forEach((line, i) => {
    if (EMAIL_RE.test(line) || PHONE_RE.test(line)) skip[i] = true;
  });
  lines.forEach((line, i) => {
    if (SIGNOFF_RE.test(line)) skip[i] = true;
  });
  for (let i = 0; i < lines.length - 1; i++) {
    if (SIGNOFF_RE.test(lines[i])) {
      const nextIsLast = i + 1 === lines.length - 1;
      const nextIsShort = lines[i + 1].split(/\s+/).length <= 3;
      if (nextIsLast && nextIsShort) skip[i + 1] = true;
    }
  }

  const content = [];
  for (let i = lines.length - 1; i >= 0 && content.length < 2; i--) {
    if (!skip[i]) content.push(lines[i]);
  }
  // content[0] is the line closest to the bottom (organisation),
  // content[1] is the one above it (name).
  const organisation = content[0] || null;
  const name = content[1] || null;

  return { name, organisation, email, phone };
}

// ---------------------------------------------------------------------
// Calendar parsing (mirrors n8n/code/06_parse_calendar.js)
// ---------------------------------------------------------------------

const ATTENDEE_LINE_RE = /^[-*]\s*(.+?),\s*(.+)$/;

function parseCalendar(events) {
  const results = [];
  for (const event of events) {
    const title = event.summary || "";
    if (!title.includes("Atlas AI Summit")) continue;
    const lines = (event.description || "").split(/\r?\n/);
    for (const line of lines) {
      const match = line.trim().match(ATTENDEE_LINE_RE);
      if (!match) continue;
      results.push({
        __kind: "calendar",
        name: match[1].trim(),
        organisation: match[2].trim(),
        email: null,
        phone: null,
        extractionStatus: "calendar_parsed",
        attempts: 1,
        sources: [`Event: ${title}`],
        flags: [],
      });
    }
  }
  return results;
}

// ---------------------------------------------------------------------
// Normalise (mirrors n8n/code/07_normalise.js and 08_tag_directory.js)
// ---------------------------------------------------------------------

function collapseWhitespace(s) {
  return s ? String(s).trim().replace(/\s+/g, " ") : null;
}

function normalisePhone(raw) {
  if (!raw) return null;
  let digits = String(raw).replace(/[^\d+]/g, "");
  if (digits.startsWith("+27")) return "+27" + digits.slice(3).replace(/\D/g, "");
  if (digits.startsWith("27") && digits.length >= 11) return "+" + digits;
  if (digits.startsWith("0")) return "+27" + digits.slice(1);
  if (digits.startsWith("+")) return digits;
  if (digits.length > 0) return "+27" + digits;
  return null;
}

function squash(s) {
  return s ? String(s).toLowerCase().replace(/[^a-z0-9]/g, "") : null;
}

function normaliseCandidate(c) {
  const name = collapseWhitespace(c.name);
  const organisation = collapseWhitespace(c.organisation);
  return {
    ...c,
    name,
    organisation,
    email: c.email ? String(c.email).trim() : null,
    phone: c.phone ? String(c.phone).trim() : null,
    normalisedName: name ? name.toLowerCase() : null,
    normalisedOrg: organisation ? organisation.toLowerCase() : null,
    squashedOrg: squash(organisation),
    normalisedEmail: c.email ? String(c.email).trim().toLowerCase() : null,
    normalisedPhone: normalisePhone(c.phone),
  };
}

function tagDirectory(speakers) {
  return speakers.map((speaker) => {
    const name = collapseWhitespace(speaker.full_name);
    const organisation = collapseWhitespace(speaker.organisation);
    return {
      __kind: "directory",
      speaker_id: speaker.speaker_id,
      name,
      organisation,
      email: speaker.email || null,
      phone: speaker.phone || null,
      normalisedName: name ? name.toLowerCase() : null,
      normalisedOrg: organisation ? organisation.toLowerCase() : null,
      squashedOrg: squash(organisation),
      normalisedEmail: speaker.email ? String(speaker.email).trim().toLowerCase() : null,
      normalisedPhone: normalisePhone(speaker.phone),
    };
  });
}

// ---------------------------------------------------------------------
// Verify & Dedupe (mirrors n8n/code/09_verify_and_dedupe.js)
// ---------------------------------------------------------------------

const NAME_MATCH_THRESHOLD = 2;
const ORG_MATCH_THRESHOLD = 2;

function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a) return b ? b.length : 0;
  if (!b) return a.length;
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

function verifyAndDedupe(candidates, directory) {
  function findExact(candidate) {
    if (candidate.normalisedEmail) {
      const hit = directory.find((d) => d.normalisedEmail === candidate.normalisedEmail);
      if (hit) return { speaker: hit, via: "exact_email" };
    }
    if (candidate.normalisedPhone) {
      const hit = directory.find((d) => d.normalisedPhone === candidate.normalisedPhone);
      if (hit) return { speaker: hit, via: "exact_phone" };
    }
    return null;
  }

  function bestNameMatch(candidate) {
    if (!candidate.normalisedName) return null;
    let best = null;
    let bestDist = Infinity;
    for (const speaker of directory) {
      const dist = levenshtein(candidate.normalisedName, speaker.normalisedName || "");
      if (dist < bestDist) {
        bestDist = dist;
        best = speaker;
      }
    }
    if (best && bestDist <= NAME_MATCH_THRESHOLD) return { speaker: best, distance: bestDist };
    return null;
  }

  function orgMatchSet(candidate) {
    if (!candidate.squashedOrg) return [];
    let bestDist = Infinity;
    for (const speaker of directory) {
      const dist = levenshtein(candidate.squashedOrg, speaker.squashedOrg || "");
      if (dist < bestDist) bestDist = dist;
    }
    if (bestDist > ORG_MATCH_THRESHOLD) return [];
    return directory.filter(
      (speaker) => levenshtein(candidate.squashedOrg, speaker.squashedOrg || "") === bestDist
    );
  }

  function resolveOne(candidate) {
    const changes = [];
    const flags = Array.isArray(candidate.flags) ? [...candidate.flags] : [];

    if (candidate.extractionStatus === "extraction_failed") {
      return { ...candidate, status: "extraction_failed", speaker_id: null, candidates: [], changes, flags };
    }

    const exact = findExact(candidate);
    const nameMatch = bestNameMatch(candidate);
    const orgSet = orgMatchSet(candidate);
    const orgIds = orgSet.map((s) => s.speaker_id);

    let speaker = null;
    let status = null;
    let ambiguousCandidates = [];

    if (exact) {
      speaker = exact.speaker;
      status = "confirmed";
    } else if (nameMatch && orgIds.length > 0 && !orgIds.includes(nameMatch.speaker.speaker_id)) {
      status = "flagged_ambiguous";
      ambiguousCandidates = Array.from(new Set([nameMatch.speaker.speaker_id, ...orgIds]));
      flags.push(
        `name best-matches ${nameMatch.speaker.speaker_id} (${nameMatch.speaker.name}) but ` +
          `organisation best-matches ${orgIds.join("/")} - not auto-resolved`
      );
    } else if (nameMatch) {
      speaker = nameMatch.speaker;
      status = nameMatch.distance === 0 ? "confirmed" : "corrected";
    } else if (!candidate.normalisedName && orgIds.length === 1) {
      speaker = orgSet[0];
      status = "corrected";
    }

    if (status === "flagged_ambiguous") {
      return { ...candidate, status, speaker_id: null, candidates: ambiguousCandidates, changes, flags };
    }

    if (!speaker) {
      return { ...candidate, status: "new_delegate", speaker_id: null, candidates: [], changes, flags };
    }

    let resultName = candidate.name;
    if (candidate.name && speaker.name && candidate.name !== speaker.name) {
      changes.push(`full_name: '${candidate.name}' -> '${speaker.name}'`);
      resultName = speaker.name;
      status = "corrected";
    } else if (!candidate.name) {
      resultName = speaker.name;
    }

    let resultOrg = candidate.organisation;
    if (candidate.organisation && speaker.organisation && candidate.organisation !== speaker.organisation) {
      changes.push(`organisation: '${candidate.organisation}' -> '${speaker.organisation}'`);
      resultOrg = speaker.organisation;
      status = "corrected";
    } else if (!candidate.organisation) {
      resultOrg = speaker.organisation;
    }

    let resultEmail = candidate.email;
    if (!candidate.email && speaker.email) {
      changes.push(`email: filled from directory ('${speaker.email}')`);
      resultEmail = speaker.email;
      status = "corrected";
    } else if (candidate.email && speaker.email && candidate.normalisedEmail !== speaker.normalisedEmail) {
      flags.push(`RSVP email '${candidate.email}' differs from directory email '${speaker.email}'`);
      status = "flagged_email_mismatch";
    }

    let resultPhone = candidate.phone;
    if (!candidate.phone && speaker.phone) resultPhone = speaker.phone;

    return {
      ...candidate,
      name: resultName,
      organisation: resultOrg,
      email: resultEmail,
      phone: resultPhone,
      status,
      speaker_id: speaker.speaker_id,
      candidates: [],
      changes,
      flags,
    };
  }

  const resolved = candidates.map(resolveOne);

  const groups = new Map();
  function groupKey(row) {
    if (row.speaker_id) return `speaker:${row.speaker_id}`;
    if (row.status === "flagged_ambiguous") return `ambiguous:${[...row.candidates].sort().join("+")}`;
    if (row.normalisedEmail) return `email:${row.normalisedEmail}`;
    if (row.normalisedPhone) return `phone:${row.normalisedPhone}`;
    return `new:${row.normalisedName || Math.random()}`;
  }

  for (const row of resolved) {
    const key = groupKey(row);
    if (!groups.has(key)) {
      groups.set(key, { ...row, sources: [...(row.sources || [])] });
      continue;
    }
    const existing = groups.get(key);
    existing.sources = Array.from(new Set([...(existing.sources || []), ...(row.sources || [])]));
    existing.changes = Array.from(new Set([...(existing.changes || []), ...(row.changes || [])]));
    existing.flags = Array.from(new Set([...(existing.flags || []), ...(row.flags || [])]));
    existing.name = existing.name || row.name;
    existing.organisation = existing.organisation || row.organisation;
    existing.email = existing.email || row.email;
    existing.phone = existing.phone || row.phone;
    if (existing.status === "confirmed" && row.status !== "confirmed") existing.status = row.status;
  }

  return Array.from(groups.values()).map((row) => ({
    name: row.name,
    organisation: row.organisation,
    email: row.email,
    phone: row.phone,
    speaker_id: row.speaker_id || null,
    status: row.status,
    candidates: row.candidates || [],
    sources: row.sources || [],
    changes: row.changes || [],
    flags: row.flags || [],
  }));
}

// ---------------------------------------------------------------------
// Score (mirrors n8n/code/11_score_vs_answer_key.js)
// ---------------------------------------------------------------------

function bySpeaker(attendees, id) {
  return attendees.find((a) => a.speaker_id === id);
}

function hasChange(attendee, substrings) {
  if (!attendee || !Array.isArray(attendee.changes)) return false;
  return attendee.changes.some((c) => substrings.every((s) => c.includes(s)));
}

function score(attendees) {
  const checks = [
    {
      id: "err-1-phone-format",
      pass: () => {
        const a = bySpeaker(attendees, "SP-013");
        return !!a && a.status !== "new_delegate" && a.status !== "flagged_ambiguous";
      },
    },
    {
      id: "err-2-misspelled-org",
      pass: () => {
        const a = bySpeaker(attendees, "SP-016");
        return !!a && a.organisation === "Silverline Robotics" && hasChange(a, ["organisation"]);
      },
    },
    {
      id: "err-3-missing-email",
      pass: () => {
        const a = bySpeaker(attendees, "SP-019");
        return !!a && a.email === "joseph.torres@example.org";
      },
    },
    {
      id: "err-4-duplicate-rsvp",
      pass: () => {
        const a = bySpeaker(attendees, "SP-008");
        return !!a && a.name === "Paige Williams" && (a.sources || []).length >= 2;
      },
    },
    {
      id: "err-5-identity-collision",
      pass: () =>
        attendees.some(
          (a) =>
            a.status === "flagged_ambiguous" &&
            Array.isArray(a.candidates) &&
            a.candidates.includes("SP-001") &&
            a.candidates.includes("SP-002")
        ),
    },
    {
      id: "err-6-email-mismatch",
      pass: () => {
        const a = bySpeaker(attendees, "SP-011");
        return !!a && a.status === "flagged_email_mismatch";
      },
    },
    {
      id: "err-7-not-in-directory",
      pass: () =>
        attendees.some((a) => a.status === "new_delegate" && a.name && a.name.includes("Priya Naidoo")),
    },
    {
      id: "err-8-calendar-org-typo",
      pass: () => {
        const a = bySpeaker(attendees, "SP-019");
        return !!a && a.organisation === "Horizon Cognition" && hasChange(a, ["Horizon Cognition"]);
      },
    },
  ];

  const results = checks.map((c) => ({ id: c.id, passed: c.pass() }));
  return { results, passedCount: results.filter((r) => r.passed).length, total: checks.length };
}

// ---------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------

console.log("getEmailText() against raw Gmail API fixtures:");

const nestedMultipartResult = getEmailText(RAW_GMAIL_FIXTURE);
const nestedMultipartOk =
  nestedMultipartResult.source === "payload.parts(text/plain)" &&
  nestedMultipartResult.text === RAW_GMAIL_FIXTURE_PLAIN_TEXT;
console.log(
  `  [${nestedMultipartOk ? "PASS" : "FAIL"}] nested multipart/mixed -> multipart/alternative -> ` +
    `text/plain (source=${nestedMultipartResult.source})`
);

const snippetFallbackResult = getEmailText(RAW_GMAIL_FIXTURE_SNIPPET_ONLY);
const snippetFallbackOk =
  snippetFallbackResult.source === "snippet" &&
  snippetFallbackResult.text === RAW_GMAIL_FIXTURE_SNIPPET_ONLY.snippet;
console.log(
  `  [${snippetFallbackOk ? "PASS" : "FAIL"}] no decodable body falls back to snippet ` +
    `(source=${snippetFallbackResult.source})`
);

if (!nestedMultipartOk || !snippetFallbackOk) {
  console.error("\ngetEmailText() fixture tests FAILED - stopping before the rest of the harness.");
  process.exit(1);
}

const speakers = JSON.parse(readFileSync(join(ROOT, "data", "kb", "speakers.json"), "utf8"));

const emailCandidates = EMAILS.map((e, i) => {
  const extracted = mockExtract(e.subject, e.body);
  return {
    __kind: "email",
    emailIndex: i + 1,
    ...extracted,
    extractionStatus: "extracted_first_try",
    attempts: 1,
    sources: [`Email ${i + 1}`],
    flags: [],
  };
});

const calendarCandidates = parseCalendar(CALENDAR_EVENTS);

const normalised = [...emailCandidates, ...calendarCandidates].map(normaliseCandidate);
const directory = tagDirectory(speakers);

const attendees = verifyAndDedupe(normalised, directory);
const { results, passedCount, total } = score(attendees);

console.log("Mock-extracted email candidates:");
for (const c of emailCandidates) {
  console.log(`  Email ${c.emailIndex}: name=${c.name}, org=${c.organisation}, email=${c.email}, phone=${c.phone}`);
}

console.log("\nFinal attendee list:");
for (const a of attendees) {
  console.log(
    `  ${a.name} | ${a.organisation} | ${a.email} | ${a.speaker_id || "(none)"} | ${a.status} | sources=${a.sources.join(", ")}`
  );
  if (a.changes.length) console.log(`      changes: ${a.changes.join(" | ")}`);
  if (a.flags.length) console.log(`      flags: ${a.flags.join(" | ")}`);
}

console.log(`\nAttendee count: ${attendees.length}`);
console.log("\nPer-error results:");
for (const r of results) {
  console.log(`  [${r.passed ? "PASS" : "FAIL"}] ${r.id}`);
}
console.log(`\nPlanted errors caught: ${passedCount} / ${total}`);
