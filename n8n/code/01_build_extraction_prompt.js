// Code node: "Build Extraction Prompt"
// Runs once per Gmail message. Live-run diagnosis: with Simplify OFF, the
// Gmail node returns the raw Gmail API message resource (id, threadId,
// labelIds, snippet, payload with base64url-encoded parts) - there is no
// top-level "text" field. The old body-decoding here missed that shape (or
// missed some variant of it), so every prompt went out with an empty body,
// which is why every Anthropic call had the same (~223) input_tokens and
// came back all-null. getEmailText() below tries several shapes in order
// and always records which one actually worked, in body_source, so this
// is debuggable directly from the node's output next time.
//
// Deliberately: the prompt carries ONLY the decoded body text and subject,
// never the Gmail "From" header or any sender metadata. That's a
// structural choice - the model has no way to use the From header for
// identity because it is never given to it. Identity must come from the
// body/signature alone, matching how the seed data
// (data/synthetic/part2_seed.md) was built: every email is actually sent
// from the same Gmail account.

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

// Recursively searches a Gmail API payload (and nested multipart parts,
// e.g. multipart/mixed -> multipart/alternative -> text/plain) for the
// first part whose mimeType matches and whose body.data is non-empty.
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

// Tries, in order: item.text, item.textPlain, a recursive search of
// item.payload.parts for text/plain, item.payload.body.data, a recursive
// search for text/html (tags stripped), then item.snippet as a last
// resort. Returns { text, source } so the source can be recorded on the
// item for debugging.
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
  const { text: bodyText, source: bodySource } = getEmailText(message);
  const subject = getHeader(message.payload, "Subject");

  return {
    json: {
      __kind: "email",
      emailIndex: index + 1,
      messageId: message.id || null,
      subject,
      bodyText,
      body_source: bodySource,
      attempt: 1,
      prompt: buildPrompt(subject, bodyText),
    },
  };
});
