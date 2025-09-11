
import React, { useState, useCallback, useRef } from 'react';
import { Header } from './components/Header.tsx';
import { CodeInput } from './components/CodeInput.tsx';
import { ResultsView } from './components/ResultsView.tsx';
import { getCodeReview } from './services/geminiService.ts';
import type { ReviewFeedback } from './types.ts';
import { DEFAULT_LANGUAGE } from './constants.ts';

export const App: React.FC = () => {
  const [code, setCode] = useState<string>('');
  const [language, setLanguage] = useState<string>(DEFAULT_LANGUAGE);
  const [review, setReview] = useState<ReviewFeedback | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  // FIX: Per coding guidelines, the API key is handled by the API service, not in the UI component.
  // This change also resolves the TypeScript error on 'import.meta.env'.

  const requestCancelled = useRef(false);

  const handleCancelRequest = useCallback(() => {
    requestCancelled.current = true;
    setIsLoading(false);
    setError('Code review cancelled by user.');
    setReview(null);
  }, []);

  const handleReviewRequest = useCallback(async () => {
    if (!code.trim()) {
      setError('Please enter some code to review.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setReview(null);
    requestCancelled.current = false; // Reset cancellation flag

    try {
      const result = await getCodeReview(code, language);
      if (requestCancelled.current) {
          console.log('Review was cancelled, discarding results.');
          return;
      }
      setReview(result);
    } catch (err) {
      if (requestCancelled.current) {
          console.log('Review was cancelled, discarding error.');
          return;
      }
      console.error(err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  }, [code, language]);

  return (
    <div className="min-h-screen bg-dark-navy text-slate font-sans">
      <Header />
      <main className="p-4 md:p-8 lg:p-12">
        <div className="max-w-screen-2xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          <div className="lg:sticky lg:top-24">
            <CodeInput
              code={code}
              setCode={setCode}
              language={language}
              setLanguage={setLanguage}
              onSubmit={handleReviewRequest}
              isLoading={isLoading}
              onCancel={handleCancelRequest}
            />
          </div>
          <div className="min-h-[500px] lg:min-h-0">
            <ResultsView
              review={review}
              isLoading={isLoading}
              error={error}
              code={code}
              language={language}
            />
          </div>
        </div>
      </main>
    </div>
  );
};