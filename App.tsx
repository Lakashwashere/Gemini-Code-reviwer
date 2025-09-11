import React, { useState, useCallback } from 'react';
import { Header } from './components/Header';
import { CodeInput } from './components/CodeInput';
import { ResultsView } from './components/ResultsView';
import { getCodeReview } from './services/geminiService';
import type { ReviewFeedback } from './types';
import { PROGRAMMING_LANGUAGES } from './constants';

const App: React.FC = () => {
  const [code, setCode] = useState<string>('');
  const [language, setLanguage] = useState<string>(PROGRAMMING_LANGUAGES[0]);
  const [review, setReview] = useState<ReviewFeedback | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const API_KEY = process.env.API_KEY;

  const handleReviewRequest = useCallback(async () => {
    if (!code.trim()) {
      setError('Please enter some code to review.');
      return;
    }

    if (!API_KEY) {
      setError('API Key is not configured. Please ensure the API_KEY environment variable is set.');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setReview(null);

    try {
      const result = await getCodeReview(code, language, API_KEY);
      setReview(result);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  }, [code, language, API_KEY]);

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
            />
          </div>
          <div className="min-h-[500px] lg:min-h-0">
            <ResultsView
              review={review}
              isLoading={isLoading}
              error={error}
              code={code}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;