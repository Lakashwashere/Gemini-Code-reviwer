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
    
    // This ref can be used in the future to cancel API requests if the service supports it.
    const abortControllerRef = useRef<AbortController | null>(null);
    
    const handleReviewRequest = useCallback(async () => {
        if (!code.trim()) {
            setError('Please enter some code to review.');
            return;
        }
        
        abortControllerRef.current = new AbortController();
        setIsLoading(true);
        setError(null);
        setReview(null);
        
        try {
            const result = await getCodeReview(code);
            setReview(result);
        } catch (err) {
            if (err instanceof Error && err.name !== 'AbortError') {
                 setError(err instanceof Error ? err.message : 'An unknown error occurred.');
            }
        } finally {
            setIsLoading(false);
            abortControllerRef.current = null;
        }
    }, [code, language]);
    
    const handleCancel = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            setIsLoading(false);
            setError("Review request cancelled.");
        }
    };
    
    return (
        <div className="min-h-screen bg-dark-navy text-light-slate font-sans flex flex-col">
            <Header />
            <main className="flex-grow p-4 md:p-6 lg:p-8 overflow-hidden">
                <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8" style={{height: 'calc(100vh - 100px)'}}>
                    <CodeInput 
                        code={code}
                        setCode={setCode}
                        language={language}
                        setLanguage={setLanguage}
                        onSubmit={handleReviewRequest}
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
