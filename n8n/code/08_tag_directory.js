// Code node: "Tag Directory"
// Input: speakers.json fetched by the HTTP Request node. n8n's HTTP
// Request node splits a top-level JSON array response into one item per
// array element automatically, so each item here is already one speaker
// record. This just tags them and adds the same normalised fields used
// for matching elsewhere, so "Verify & Dedupe" can compare like with like.

function collapseWhitespace(s) {
  return s ? String(s).trim().replace(/\s+/g, " ") : null;
}

function normalisePhone(raw) {
  if (!raw) return null;
  let digits = String(raw).replace(/[^\d+]/g, "");
  if (digits.startsWith("+27")) {
    return "+27" + digits.slice(3).replace(/\D/g, "");
  }
  if (digits.startsWith("27") && digits.length >= 11) {
    return "+" + digits;
  }
  if (digits.startsWith("0")) {
    return "+27" + digits.slice(1);
  }
  if (digits.startsWith("+")) {
    return digits;
  }
  if (digits.length > 0) {
    return "+27" + digits;
  }
  return null;
}

function squash(s) {
  return s ? String(s).toLowerCase().replace(/[^a-z0-9]/g, "") : null;
}

return items.map((item) => {
  const speaker = item.json;
  const name = collapseWhitespace(speaker.full_name);
  const organisation = collapseWhitespace(speaker.organisation);

  return {
    json: {
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
    },
  };
});
