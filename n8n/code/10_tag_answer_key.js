// Code node: "Tag Answer Key"
// Input: the single JSON object fetched from part2_answer_key.json. Just
// tags it so it survives the merge into the "Score vs Answer Key" node
// alongside the attendee rows.

return items.map((item) => ({
  json: {
    __kind: "answer_key",
    answerKey: item.json,
  },
}));
