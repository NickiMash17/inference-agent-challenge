// Code node: "Validate & Shape (Retry)"
// Same logic as "Validate & Shape (Attempt 1)", but pulls sibling fields
// from "Build Retry Prompt" instead, since this validates the SECOND
// (retry) extraction call's response.
//
// Live-run diagnosis: reading `content[0].text` returned "empty response
// text" even though a text block was present in `content`, because the
// first content block isn't always the text block (a thinking/redacted
// block can come first). Fixed by filtering content for type === "text"
// and joining it, instead of indexing position 0.

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function getResponseText(response) {
  const blocks = Array.isArray(response.content) ? response.content : [];
  return blocks
    .filter((c) => c && c.type === "text")
    .map((c) => c.text)
    .join("");
}

function parseClaudeJson(text) {
  if (!text) return { ok: false, reason: "empty response text" };
  let cleaned = text.trim();
  const fenceMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenceMatch) cleaned = fenceMatch[1].trim();
  try {
    return { ok: true, value: JSON.parse(cleaned) };
  } catch (err) {
    return { ok: false, reason: `invalid JSON: ${err.message}` };
  }
}

function validate(extracted) {
  if (!extracted || typeof extracted !== "object") {
    return { valid: false, reason: "extraction is not a JSON object" };
  }
  if (!extracted.name && !extracted.organisation && !extracted.email && !extracted.phone) {
    return { valid: false, reason: "no fields extracted" };
  }
  if (!extracted.name || typeof extracted.name !== "string" || !extracted.name.trim()) {
    return { valid: false, reason: "name is missing or empty" };
  }
  if (extracted.email && !EMAIL_RE.test(String(extracted.email).trim())) {
    return { valid: false, reason: `email "${extracted.email}" is not a valid email format` };
  }
  return { valid: true, reason: null };
}

const results = [];
for (let i = 0; i < items.length; i++) {
  const response = items[i].json;
  const prior = $("Build Retry Prompt").all()[i].json;

  const textBlock = getResponseText(response);
  const parsed = parseClaudeJson(textBlock);

  let valid = false;
  let reason = null;
  let extracted = null;

  if (!parsed.ok) {
    reason = parsed.reason;
  } else {
    extracted = parsed.value;
    const check = validate(extracted);
    valid = check.valid;
    reason = check.reason;
  }

  results.push({
    json: {
      __kind: "email",
      emailIndex: prior.emailIndex,
      messageId: prior.messageId,
      subject: prior.subject,
      bodyText: prior.bodyText,
      body_source: prior.body_source,
      attempt: prior.attempt,
      valid,
      reason,
      name: extracted ? extracted.name || null : null,
      organisation: extracted ? extracted.organisation || null : null,
      email: extracted ? extracted.email || null : null,
      phone: extracted ? extracted.phone || null : null,
      extractionStatus: valid ? "extracted_after_retry" : "extraction_failed",
      sources: [`Email ${prior.emailIndex}`],
    },
  });
}

return results;
