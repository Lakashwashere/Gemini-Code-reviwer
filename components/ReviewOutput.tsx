import React from 'react';
import type { ReviewFeedback, Suggestion } from '../types.ts';

interface ReviewOutputProps {
  review: ReviewFeedback;
}

const getCategoryStyle = (category: string) => {
  switch (category.toLowerCase()) {
    case 'performance': return 'text-purple-300 bg-purple-500/10 border-purple-500/30';
    case 'security': return 'text-red-300 bg-red-500/10 border-red-500/30';
    case 'readability': return 'text-blue-300 bg-blue-500/10 border-blue-500/30';
    case 'best practice': return 'text-accent bg-accent/10 border-accent/30';
    case 'logic': return 'text-yellow-300 bg-yellow-500/10 border-yellow-500/30';
    default: return 'text-slate bg-light-navy/50 border-light-navy';
  }
};

const SuggestionCard: React.FC<{ suggestion: Suggestion }> = ({ suggestion }) => {
  return (
    <div className="bg-light-navy rounded-lg p-4 border border-light-navy hover:border-accent/50 transition-colors duration-300">
      <div className="flex items-center justify-between mb-3 gap-4">
        <span className={`px-3 py-1 text-xs font-semibold font-mono rounded-full border ${getCategoryStyle(suggestion.category)}`}>
          {suggestion.category}
        </span>
        {suggestion.file && (
          <span className="text-xs font-mono text-slate truncate" title={suggestion.file}>
            {suggestion.file}
          </span>
        )}
      </div>
      <p className="text-light-slate mb-3">{suggestion.description}</p>
      <div className="bg-dark-navy rounded-md p-3 border-l-2 border-accent/50">
        <p className="font-mono text-sm text-lightest-slate whitespace-pre-wrap">
          {suggestion.suggestion}
        </p>
      </div>
    </div>
  );
};

export const ReviewOutput: React.FC<ReviewOutputProps> = ({ review }) => {
  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-2xl font-bold text-lightest-slate mb-3 relative flex items-center gap-4 after:content-[''] after:w-full after:h-[1px] after:bg-light-navy">Summary</h3>
        <p className="text-light-slate bg-light-navy/50 p-4 rounded-lg border border-light-navy">{review.summary}</p>
      </div>
      <div>
        <h3 className="text-2xl font-bold text-lightest-slate mb-4 relative flex items-center gap-4 after:content-[''] after:w-full after:h-[1px] after:bg-light-navy">Suggestions</h3>
        {review.suggestions.length > 0 ? (
            <div className="space-y-4">
              {review.suggestions.map((s, index) => (
                <SuggestionCard key={index} suggestion={s} />
              ))}
            </div>
        ) : (
            <p className="text-slate bg-light-navy/50 p-4 rounded-lg border border-light-navy">No specific suggestions. The code looks great!</p>
        )}
      </div>
    </div>
  );
};