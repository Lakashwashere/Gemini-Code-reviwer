
import type { ReviewFeedback } from '../types';
import { getMarkdownLanguageFromPath, parseMultiFileCode } from './codeParser';

export const exportReviewAsMarkdown = (review: ReviewFeedback) => {
  const { summary, suggestions, explanation, revisedCode } = review;

  let markdownContent = `# Gemini Code Review\n\n`;
  
  markdownContent += `## Summary\n\n${summary}\n\n`;

  if (suggestions.length > 0) {
      markdownContent += `## Suggestions\n\n`;
      suggestions.forEach(s => {
          markdownContent += `### ${s.category} in \`${s.file}\`\n\n`;
          markdownContent += `**Issue:** ${s.description}\n\n`;
          markdownContent += `**Suggestion:**\n> ${s.suggestion.replace(/\n/g, '\n> ')}\n\n`;
          markdownContent += `---\n\n`;
      });
  }
  
  markdownContent += `## Explanation of Changes\n\n${explanation}\n\n`;

  markdownContent += `## Revised Code\n\n`;
  const files = parseMultiFileCode(revisedCode);
  files.forEach(file => {
      const lang = getMarkdownLanguageFromPath(file.path);
      markdownContent += `### \`${file.path}\`\n\n`;
      markdownContent += `\`\`\`${lang}\n${file.content}\n\`\`\`\n\n`;
  });

  const blob = new Blob([markdownContent], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'gemini-code-review.md';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};