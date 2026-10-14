// Code node: "Mark Extraction Failed"
// Input: items still invalid after the retry (FALSE branch of "Valid?
// (Retry)"). Whatever partial fields the model produced are unreliable, so
// this clears them and shapes the item into the same candidate-row shape
// used everywhere downstream, flagged for a human to resolve.

return items.map((item) => {
  const prior = item.json;
  return {
    json: {
      __kind: "email",
      emailIndex: prior.emailIndex,
      messageId: prior.messageId,
      subject: prior.subject,
      name: null,
      organisation: null,
      email: null,
      phone: null,
      extractionStatus: "extraction_failed",
      attempts: 2,
      sources: [`Email ${prior.emailIndex}`],
      flags: [`extraction_failed: ${prior.reason}`],
    },
  };
});
