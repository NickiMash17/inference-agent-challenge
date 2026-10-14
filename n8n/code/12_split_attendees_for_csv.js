// Code node: "Split Attendees for CSV"
// Input: the passthrough output of "Score vs Answer Key" (attendee rows
// plus one score_summary row). Keeps only the attendee rows, flattens the
// array fields to plain strings so "Convert to File" produces a readable
// CSV column instead of "[object Object]".

return items
  .filter((item) => item.json.__kind === "attendee")
  .map((item) => {
    const a = item.json;
    return {
      json: {
        name: a.name,
        organisation: a.organisation,
        email: a.email,
        phone: a.phone,
        speaker_id: a.speaker_id || "",
        status: a.status,
        candidates: (a.candidates || []).join("; "),
        sources: (a.sources || []).join("; "),
        changes: (a.changes || []).join("; "),
        flags: (a.flags || []).join("; "),
      },
    };
  });
