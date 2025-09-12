import type { ReviewFeedback } from '../types.ts';

export const generateImprovementPrompt = (review: ReviewFeedback, originalCode: string): string => {
  let prompt = `You are an expert software engineer. Your task is to refactor the following code based on the provided code review feedback.

Carefully analyze the original code and the suggestions, and then provide the full, complete, and refactored code that incorporates all the suggested improvements.

RULES:
- Apply ALL suggestions from the review.
- The output should ONLY be the final, revised code.
- Do not include explanations, summaries, or any other text outside of the code itself.
- Preserve the original multi-file format ('// FILE: path/to/file.ext') if it exists.

---

## ORIGINAL CODE:
\`\`\`
${originalCode.trim()}
\`\`\`

---

## CODE REVIEW FEEDBACK:

### Summary:
${review.summary}

### Specific Suggestions:
`;

  if (review.suggestions.length > 0) {
    review.suggestions.forEach((s, index) => {
      // FIX: Safely handle the optional 'file' property to avoid printing "undefined".
      const fileInfo = s.file ? ` in file "${s.file}"` : '';
      prompt += `${index + 1}. **[${s.category}]${fileInfo}**:
   - **Issue:** ${s.description}
   - **Suggestion:** ${s.suggestion}\n\n`;
    });
  } else {
    prompt += "No specific suggestions were provided, but please review for any potential improvements based on the summary.\n";
  }

  prompt += `---

Now, provide the complete, refactored code based on these instructions.`;

  return prompt;
};
