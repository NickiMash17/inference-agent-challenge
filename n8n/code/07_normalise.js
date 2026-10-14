// Code node: "Normalise"
// Input: merged email-extraction candidates + calendar-parsed candidates.
// Adds normalised fields used for matching further down the pipeline.
// Display fields (name/organisation/email/phone) are left as extracted;
// only the normalised* fields are lowercased/reformatted.

function collapseWhitespace(s) {
  return s ? String(s).trim().replace(/\s+/g, " ") : null;
}

function normaliseEmail(email) {
  return email ? String(email).trim().toLowerCase() : null;
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
  const c = item.json;
  const name = collapseWhitespace(c.name);
  const organisation = collapseWhitespace(c.organisation);

  return {
    json: {
      ...c,
      name,
      organisation,
      email: c.email ? String(c.email).trim() : null,
      phone: c.phone ? String(c.phone).trim() : null,
      normalisedName: name ? name.toLowerCase() : null,
      normalisedOrg: organisation ? organisation.toLowerCase() : null,
      squashedOrg: squash(organisation),
      normalisedEmail: normaliseEmail(c.email),
      normalisedPhone: normalisePhone(c.phone),
    },
  };
});
