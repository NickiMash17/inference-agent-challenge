// Code node: "Build Retry Prompt"
// Input: items that failed validation after the first extraction attempt
// (the FALSE branch of the "Valid? (Attempt 1)" IF node). Builds a second
// prompt that shows the model its own invalid output and the validation
// error, and asks it to correct it. This is the visible self-correction
// step - its result is recorded by "Validate & Shape (Retry)" regardless
// of whether the retry succeeds.

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
      bodyText: prior.bodyText,
      attempt: 2,
      prompt: buildRetryPrompt(prior.subject, prior.bodyText, previousAttemptJson, prior.reason),
    },
  };
});
