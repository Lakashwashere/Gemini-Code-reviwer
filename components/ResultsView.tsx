import React, { useState } from 'react';
import type { ReviewFeedback } from '../types';
import { ReviewOutput } from './ReviewOutput';
import { RevisedCodeOutput } from './RevisedCodeOutput';
import { ExplanationOutput } from './ExplanationOutput';
import { SparklesIcon } from './icons/SparklesIcon';

interface ResultsViewProps {
  review: ReviewFeedback | null;
  isLoading: boolean;
  error: string | null;
}

type Tab = 'suggestions' | 'explanation' | 'revisedCode';

export const ResultsView: React.FC<ResultsViewProps> = ({ review, isLoading, error }) => {
  const [activeTab, setActiveTab] = useState<Tab>('suggestions');

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
        <div className="border-b border-light-navy mb-4">
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
          </nav>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto">
          {activeTab === 'suggestions' && <ReviewOutput review={review} />}
          {activeTab === 'explanation' && <ExplanationOutput explanation={review.explanation} />}
          {activeTab === 'revisedCode' && <RevisedCodeOutput code={review.revisedCode} />}
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