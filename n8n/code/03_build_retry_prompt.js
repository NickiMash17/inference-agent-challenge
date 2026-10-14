// Code node: "Build Retry Prompt"
// Input: items that failed validation after the first extraction attempt
// (the FALSE branch of the "Valid? (Attempt 1)" IF node). Builds a second
// prompt that shows the model its own invalid output and the validation
// error, and asks it to correct it. This is the visible self-correction
// step - its result is recorded by "Validate & Shape (Retry)" regardless
// of whether the retry succeeds.
//
// getEmailText() is duplicated from "Build Extraction Prompt" as a safety
// net: normally bodyText already carries through from that node via
// `prior.bodyText`, but if it's somehow still empty here, this re-derives
// it from the original Gmail message (looked up by paired item index)
// rather than silently retrying with another empty body.

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

function buildRetryPrompt(subject, bodyText, previousAttemptJson, reason) {
  return [
    "Your previous attempt to extract attendee contact details from this",
    "conference RSVP email was invalid. Fix it.",
    "",
    `Validation error: ${reason}`,
    "",
    `Your previous output was: ${previousAttemptJson}`,
    "",
    "Read ONLY the subject and body text below. Do not assume anything",
    "about who sent this email - the attendee's name, organisation, email,",
    "and phone must come only from what is written in the body/signature.",
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

return items.map((item) => {
  const prior = item.json;

  let bodyText = prior.bodyText;
  let bodySource = prior.body_source;
  if (!bodyText || !bodyText.trim()) {
    // Match by messageId, NOT by array position: this node's input is
    // already the FALSE branch of the attempt-1 IF node, a filtered
    // subset of "Gmail - Get Many"'s output, so positional indices
    // between the two no longer line up.
    const originalItem = $("Gmail - Get Many")
      .all()
      .find((it) => it.json.id === prior.messageId);
    const reExtracted = getEmailText(originalItem ? originalItem.json : {});
    bodyText = reExtracted.text;
    bodySource = `retry-reextracted:${reExtracted.source}`;
  }

  const previousAttemptJson = JSON.stringify({
    name: prior.name,
    organisation: prior.organisation,
    email: prior.email,
    phone: prior.phone,
  });

  return {
    json: {
      __kind: "email",
      emailIndex: prior.emailIndex,
      messageId: prior.messageId,
      subject: prior.subject,
      bodyText,
      body_source: bodySource,
      attempt: 2,
      prompt: buildRetryPrompt(prior.subject, bodyText, previousAttemptJson, prior.reason),
    },
  };
});
