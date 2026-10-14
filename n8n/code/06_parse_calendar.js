// Code node: "Parse Calendar"
// Deterministic parsing (no LLM) of Google Calendar event descriptions.
// Expects events whose description contains lines shaped like
// "- Name, Organisation" (see data/synthetic/part2_seed.md). Only keeps
// events whose title actually contains "Atlas AI Summit" - the Calendar
// node's own query narrows this already, but this is the one place that
// deterministically enforces it regardless of how loose that query match
// is.

const ATTENDEE_LINE_RE = /^[-*]\s*(.+?),\s*(.+)$/;

const results = [];

items.forEach((item, eventIndex) => {
  const event = item.json;
  const title = event.summary || event.title || "";
  if (!title.includes("Atlas AI Summit")) return;

  const description = event.description || "";
  const lines = description.split(/\r?\n/);

  lines.forEach((line) => {
    const trimmed = line.trim();
    const match = trimmed.match(ATTENDEE_LINE_RE);
    if (!match) return;

    const name = match[1].trim();
    const organisation = match[2].trim();

    results.push({
      json: {
        __kind: "calendar",
        name,
        organisation,
        email: null,
        phone: null,
        extractionStatus: "calendar_parsed",
        attempts: 1,
        sources: [`Event: ${title}`],
        flags: [],
      },
    });
  });
});

return results;
