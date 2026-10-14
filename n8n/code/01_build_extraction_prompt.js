// Code node: "Build Extraction Prompt"
// Runs once per Gmail message (Gmail node, Simplify OFF -> raw Gmail API
// message resource). Decodes the plain-text body and builds the prompt for
// the first extraction attempt against the Anthropic Messages API.
//
// Deliberately: the prompt below carries ONLY the decoded body text and
// subject, never the Gmail "From" header or any sender metadata. That's a
// structural choice, not just an instruction to the model - the model has
// no way to use the From header for identity because it is never given to
// it. Identity must come from the body/signature alone, which matches how
// the seed data (data/synthetic/part2_seed.md) was built: every email is
// actually sent from the same Gmail account.

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

function findPart(payload, mimeType) {
  if (!payload) return null;
  if (payload.mimeType === mimeType && payload.body && payload.body.data) {
    return payload.body.data;
  }
  if (Array.isArray(payload.parts)) {
    for (const part of payload.parts) {
      const found = findPart(part, mimeType);
      if (found) return found;
    }
  }
  return null;
}

function getBodyText(payload) {
  const plain = findPart(payload, "text/plain");
  if (plain) return decodeBase64Url(plain).trim();

  const html = findPart(payload, "text/html");
  if (html) return stripHtml(decodeBase64Url(html));

  if (payload && payload.body && payload.body.data) {
    return decodeBase64Url(payload.body.data).trim();
  }
  return "";
}

function getHeader(payload, name) {
  const headers = (payload && payload.headers) || [];
  const header = headers.find(
    (h) => h.name && h.name.toLowerCase() === name.toLowerCase()
  );
  return header ? header.value : "";
}

function buildPrompt(subject, bodyText) {
  return [
    "You are extracting attendee contact details from a single conference",
    "RSVP email for the Atlas AI Summit.",
    "",
    "Read ONLY the subject and body text below. Do not assume anything",
    "about who sent this email - there is no reliable sender identity",
    "available to you, so the attendee's name, organisation, email, and",
    "phone must come only from what is written in the body/signature.",
    "",
    `Subject: ${subject}`,
    "Body:",
    "---",
    bodyText,
    "---",
    "",
    "Respond with ONLY a JSON object, no other text, no markdown code",
    "fences, in exactly this shape:",
    '{"name": string or null, "organisation": string or null, ' +
      '"email": string or null, "phone": string or null}',
    "",
    "Use null for any field not stated in the body. Do not invent values.",
  ].join("\n");
}

return items.map((item, index) => {
  const message = item.json;
  const bodyText = getBodyText(message.payload);
  const subject = getHeader(message.payload, "Subject");

  return {
    json: {
      __kind: "email",
      emailIndex: index + 1,
      messageId: message.id || null,
      subject,
      bodyText,
      attempt: 1,
      prompt: buildPrompt(subject, bodyText),
    },
  };
});
