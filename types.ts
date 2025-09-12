export interface Suggestion {
  category: 'Readability' | 'Performance' | 'Security' | 'Best Practice' | 'Logic' | string;
  description: string;
  suggestion: string;
  // FIX: Add optional 'file' property to align with usage in components.
  file?: string;
}

export interface ReviewFeedback {
  summary: string;
  suggestions: Suggestion[];
  // FIX: Add 'explanation' property as it is used in various components.
  explanation: string;
  revisedCode: string;
}
