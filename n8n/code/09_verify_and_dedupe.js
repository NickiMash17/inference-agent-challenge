// Code node: "Verify & Dedupe"
// Input: normalised candidate rows (__kind 'email'/'calendar') merged with
// the tagged speaker directory (__kind 'directory'). Resolves each
// candidate against the directory, then dedupes resolved candidates into
// one row per real attendee. Outputs __kind 'attendee' rows.

const NAME_MATCH_THRESHOLD = 2; // max Levenshtein edits to call two names "the same"
const ORG_MATCH_THRESHOLD = 2; // max Levenshtein edits on squashed org strings

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
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[m][n];
}

const all = items.map((i) => i.json);
const directory = all.filter((i) => i.__kind === "directory");
const candidates = all.filter((i) => i.__kind === "email" || i.__kind === "calendar");

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
    return {
      ...candidate,
      status: "extraction_failed",
      speaker_id: null,
      candidates: [],
      changes,
      flags,
    };
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
    // Best name match and best org match disagree on WHO this is.
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
    return {
      ...candidate,
      status,
      speaker_id: null,
      candidates: ambiguousCandidates,
      changes,
      flags,
    };
  }

  if (!speaker) {
    return {
      ...candidate,
      status: "new_delegate",
      speaker_id: null,
      candidates: [],
      changes,
      flags,
    };
  }

  // Identity resolved to `speaker` - reconcile field-level differences.
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
  if (!candidate.phone && speaker.phone) {
    resultPhone = speaker.phone;
  }

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

// Dedupe: group by speaker_id, else by ambiguous-candidate-set, else by
// normalised email/phone, else treat as a distinct new delegate.
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
  // Prefer a non-null value already on the group; otherwise take this row's.
  existing.name = existing.name || row.name;
  existing.organisation = existing.organisation || row.organisation;
  existing.email = existing.email || row.email;
  existing.phone = existing.phone || row.phone;
  // "corrected"/"flagged_*" outrank a plain "confirmed" when merging.
  if (existing.status === "confirmed" && row.status !== "confirmed") {
    existing.status = row.status;
  }
}

return Array.from(groups.values()).map((row) => ({
  json: {
    __kind: "attendee",
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
  },
}));
