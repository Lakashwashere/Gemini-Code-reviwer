import React, { useState, useMemo } from 'react';
import type { ReviewFeedback } from '../types';
import { ReviewOutput } from './ReviewOutput';
import { RevisedCodeOutput } from './RevisedCodeOutput';
import { ExplanationOutput } from './ExplanationOutput';
import { AIPromptOutput } from './AIPromptOutput';
import { SparklesIcon } from './icons/SparklesIcon';
import { DownloadIcon } from './icons/DownloadIcon';
import { getMarkdownLanguageFromPath, parseMultiFileCode } from '../utils/codeParser';
import { generateImprovementPrompt } from '../utils/promptGenerator';

interface ResultsViewProps {
  review: ReviewFeedback | null;
  isLoading: boolean;
  error: string | null;
  code: string;
}

type Tab = 'suggestions' | 'explanation' | 'revisedCode' | 'aiPrompt';

export const ResultsView: React.FC<ResultsViewProps> = ({ review, isLoading, error, code }) => {
  const [activeTab, setActiveTab] = useState<Tab>('suggestions');

  const aiPrompt = useMemo(() => {
    if (!review) return '';
    return generateImprovementPrompt(review, code);
  }, [review, code]);

  const handleSaveReview = () => {
    if (!review) return;

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

  if (isLoading) {
    return (
      <div className="bg-navy rounded-lg shadow-lg p-6 flex flex-col items-center justify-center h-full text-center">
        <SparklesIcon className="h-16 w-16 text-accent animate-pulse-fast mb-4" />
        <h3 className="text-xl font-semibold text-lightest-slate">Analyzing Your Code...</h3>
        <p className="text-slate mt-2">Gemini is working its magic. This might take a moment.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-navy rounded-lg shadow-lg p-6 flex items-center justify-center h-full text-center border border-red-500/50">
        <div>
          <h3 className="text-xl font-semibold text-red-300">An Error Occurred</h3>
          <p className="text-red-400 mt-2">{error}</p>
        </div>
      </div>
    );
  }
  
  if (review) {
    return (
      <div className="bg-navy rounded-lg shadow-lg p-6 flex flex-col h-full">
        <div className="border-b border-light-navy mb-4 flex justify-between items-center">
          <nav className="-mb-px flex space-x-6" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('suggestions')}
              className={`${
                activeTab === 'suggestions'
                  ? 'border-accent text-accent'
                  : 'border-transparent text-slate hover:text-lightest-slate hover:border-slate'
              } whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200 focus:outline-none`}
              aria-current={activeTab === 'suggestions' ? 'page' : undefined}
            >
              Suggestions
            </button>
            <button
              onClick={() => setActiveTab('explanation')}
              className={`${
                activeTab === 'explanation'
                  ? 'border-accent text-accent'
                  : 'border-transparent text-slate hover:text-lightest-slate hover:border-slate'
              } whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200 focus:outline-none`}
              aria-current={activeTab === 'explanation' ? 'page' : undefined}
            >
              Explanation
            </button>
            <button
              onClick={() => setActiveTab('revisedCode')}
              className={`${
                activeTab === 'revisedCode'
                  ? 'border-accent text-accent'
                  : 'border-transparent text-slate hover:text-lightest-slate hover:border-slate'
              } whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200 focus:outline-none`}
              aria-current={activeTab === 'revisedCode' ? 'page' : undefined}
            >
              Revised Code
            </button>
            <button
              onClick={() => setActiveTab('aiPrompt')}
              className={`${
                activeTab === 'aiPrompt'
                  ? 'border-accent text-accent'
                  : 'border-transparent text-slate hover:text-lightest-slate hover:border-slate'
              } whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200 focus:outline-none`}
              aria-current={activeTab === 'aiPrompt' ? 'page' : undefined}
            >
              AI Prompt
            </button>
          </nav>
          <button
            onClick={handleSaveReview}
            className="flex items-center text-sm font-medium text-slate hover:text-accent transition-colors duration-200 p-2 rounded-md"
            aria-label="Save review as Markdown"
            title="Save review as Markdown"
          >
            <DownloadIcon className="h-5 w-5 mr-2" />
            Save Review
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto">
          {activeTab === 'suggestions' && <ReviewOutput review={review} />}
          {activeTab === 'explanation' && <ExplanationOutput explanation={review.explanation} />}
          {activeTab === 'revisedCode' && <RevisedCodeOutput code={review.revisedCode} />}
          {activeTab === 'aiPrompt' && <AIPromptOutput prompt={aiPrompt} />}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-navy rounded-lg shadow-lg p-6 flex flex-col items-center justify-center h-full text-center border-2 border-dashed border-light-navy">
      <h3 className="text-xl font-semibold text-slate">Ready for Review</h3>
      <p className="text-slate/70 mt-2">Your code review results will appear here once you submit your code.</p>
    </div>
  );
};