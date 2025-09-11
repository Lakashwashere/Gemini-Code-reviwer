
import React from 'react';
import { CodeIcon } from './icons/CodeIcon';

export const Header: React.FC = () => {
  return (
    <header className="bg-dark-navy/80 backdrop-blur-lg p-4 border-b border-light-navy sticky top-0 z-10">
      <div className="max-w-6xl mx-auto flex items-center space-x-3">
        <CodeIcon className="h-8 w-8 text-accent" />
        <h1 className="text-2xl font-bold text-lightest-slate tracking-tight">
          Gemini Code Reviewer
        </h1>
      </div>
    </header>
  );
};