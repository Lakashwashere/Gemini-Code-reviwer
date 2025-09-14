import React, { useState, useRef, useCallback } from 'react';
import { CodeInput } from './components/CodeInput.tsx';
import { ResultsView } from './components/ResultsView.tsx';
import { Header } from './components/Header.tsx';
import { getCodeReview } from './services/geminiService.ts';
import type { ReviewFeedback } from './types.ts';
import { DEFAULT_LANGUAGE } from './constants.ts';

export const App: React.FC = () => {
  const [code, setCode] = useState<string>('');
  const [language, setLanguage] = useState<string>(DEFAULT_LANGUAGE);
  const [review, setReview] = useState<ReviewFeedback | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Use a ref to store the AbortController
  const cancellationController = useRef<AbortController | null>(null);

  const handleSubmit = useCallback(async () => {
    if (!code.trim()) {
      setError("Code input cannot be empty.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setReview(null);
    
    // Create a new AbortController for this request
    cancellationController.current = new AbortController();

    try {
      const result = await getCodeReview(code);
      setReview(result);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('The code review was cancelled.');
        console.log('API request was cancelled by the user.');
      } else {
        const errorMessage = err instanceof Error ? err.message : "An unknown error occurred.";
        setError(errorMessage);
        console.error("Error during code review:", err);
      }
    } finally {
      setIsLoading(false);
      cancellationController.current = null; // Clear the controller
    }
  }, [code]);
  
  const handleCancel = useCallback(() => {
    if (cancellationController.current) {
      cancellationController.current.abort();
      console.log('Cancellation signal sent.');
    }
  }, []);

  return (
    <div className="min-h-screen bg-navy flex flex-col">
      <Header />
      <main className="flex-grow p-4 md:p-6 lg:p-8" style={{height: 'calc(100vh - 72px)'}}>
        <div className="max-w-screen-2xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
          <CodeInput
            code={code}
            setCode={setCode}
            language={language}
            setLanguage={setLanguage}
            onSubmit={handleSubmit}
            isLoading={isLoading}
            onCancel={handleCancel}
          />
          <ResultsView
            review={review}
            isLoading={isLoading}
            error={error}
            code={code}
            language={language}
          />
        </div>
      </main>
    </div>
  );
};
