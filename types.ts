export interface Suggestion {
  file?: string;
  category: 'Readability' | 'Performance' | 'Security' | 'Best Practice' | 'Logic' | string;
  description: string;
  suggestion: string;
}

export interface ReviewFeedback {
  summary: string;
  suggestions: Suggestion[];
  revisedCode: string;
  explanation: string;
}
