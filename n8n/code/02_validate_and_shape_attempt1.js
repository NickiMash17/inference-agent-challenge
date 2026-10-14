// Code node: "Validate & Shape (Attempt 1)"
// Input: the Anthropic Messages API response for the FIRST extraction call.
// Pulls sibling fields (subject, bodyText, emailIndex, attempt) back from
// the "Build Extraction Prompt" node by name, since the HTTP Request node
// in between replaces $json with the raw API response. n8n preserves 1:1
// item pairing through HTTP Request, so $('Node Name').item is safe here.

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function parseClaudeJson(text) {
  if (!text) return { ok: false, reason: "empty response text" };
  let cleaned = text.trim();
  // Strip ```json ... ``` or ``` ... ``` fences if the model added them.
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
  const prior = $("Build Extraction Prompt").all()[i].json;

  const textBlock =
    response.content && response.content[0] && response.content[0].text;
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
      attempt: prior.attempt,
      valid,
      reason,
      name: extracted ? extracted.name || null : null,
      organisation: extracted ? extracted.organisation || null : null,
      email: extracted ? extracted.email || null : null,
      phone: extracted ? extracted.phone || null : null,
      extractionStatus: valid ? "extracted_first_try" : "needs_retry",
      sources: [`Email ${prior.emailIndex}`],
    },
  });
}

return results;
