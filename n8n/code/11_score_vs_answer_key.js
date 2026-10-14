// Code node: "Score vs Answer Key"
// Input: resolved attendee rows (__kind 'attendee') merged with the tagged
// answer key (__kind 'answer_key'). Passes the attendee rows through
// unchanged and adds one extra item (__kind 'score_summary') reporting
// "Planted errors caught: X / 8" plus a per-error pass/fail breakdown.
//
// Each check below is keyed to one planted_errors[].id from
// data/synthetic/part2_answer_key.json and inspects the final attendee
// list for the specific, documented resolution that error is supposed to
// produce - it does not require exact string equality with every field in
// the answer key, since several reasonable (but not byte-identical)
// resolutions can be equally "correct" for a flagged/ambiguous case.

const all = items.map((i) => i.json);
const attendees = all.filter((i) => i.__kind === "attendee");
const answerKeyItem = all.find((i) => i.__kind === "answer_key");
const answerKey = answerKeyItem ? answerKeyItem.answerKey : null;

function bySpeaker(id) {
  return attendees.find((a) => a.speaker_id === id);
}

function hasChange(attendee, substrings) {
  if (!attendee || !Array.isArray(attendee.changes)) return false;
  return attendee.changes.some((c) => substrings.every((s) => c.includes(s)));
}

const checks = [
  {
    id: "err-1-phone-format",
    pass: () => {
      const a = bySpeaker("SP-013");
      return !!a && a.status !== "new_delegate" && a.status !== "flagged_ambiguous";
    },
  },
  {
    id: "err-2-misspelled-org",
    pass: () => {
      const a = bySpeaker("SP-016");
      return !!a && a.organisation === "Silverline Robotics" && hasChange(a, ["organisation"]);
    },
  },
  {
    id: "err-3-missing-email",
    pass: () => {
      const a = bySpeaker("SP-019");
      return !!a && a.email === "joseph.torres@example.org";
    },
  },
  {
    id: "err-4-duplicate-rsvp",
    pass: () => {
      const a = bySpeaker("SP-008");
      return !!a && a.name === "Paige Williams" && (a.sources || []).length >= 2;
    },
  },
  {
    id: "err-5-identity-collision",
    pass: () => {
      const flagged = attendees.find(
        (a) =>
          a.status === "flagged_ambiguous" &&
          Array.isArray(a.candidates) &&
          a.candidates.includes("SP-001") &&
          a.candidates.includes("SP-002")
      );
      return !!flagged;
    },
  },
  {
    id: "err-6-email-mismatch",
    pass: () => {
      const a = bySpeaker("SP-011");
      return !!a && a.status === "flagged_email_mismatch";
    },
  },
  {
    id: "err-7-not-in-directory",
    pass: () => {
      return attendees.some(
        (a) => a.status === "new_delegate" && a.name && a.name.includes("Priya Naidoo")
      );
    },
  },
  {
    id: "err-8-calendar-org-typo",
    pass: () => {
      const a = bySpeaker("SP-019");
      return !!a && a.organisation === "Horizon Cognition" && hasChange(a, ["Horizon Cognition"]);
    },
  },
];

const results = checks.map((check) => ({
  id: check.id,
  passed: check.pass(),
}));

const passedCount = results.filter((r) => r.passed).length;

const output = attendees.map((a) => ({ json: a }));
output.push({
  json: {
    __kind: "score_summary",
    summary: `Planted errors caught: ${passedCount} / ${checks.length}`,
    results,
    attendee_count: attendees.length,
    expected_attendee_count: answerKey ? answerKey.expected_final_attendee_count : null,
  },
});

return output;
