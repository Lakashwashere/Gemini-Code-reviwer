import React, { useState } from 'react';
import { CopyIcon } from './icons/CopyIcon';
import { CheckIcon } from './icons/CheckIcon';

interface AIPromptOutputProps {
  prompt: string;
}

export const AIPromptOutput: React.FC<AIPromptOutputProps> = React.memo(({ prompt }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(prompt).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(err => {
      console.error('Failed to copy prompt: ', err);
    });
  };

  return (
    <div className="space-y-4">
      <h3 className="text-2xl font-bold text-lightest-slate relative flex items-center gap-4 after:content-[''] after:w-full after:h-[1px] after:bg-light-navy">
        AI-Ready Prompt
      </h3>
      <div className="text-light-slate bg-light-navy/50 p-4 rounded-lg border border-light-navy">
        <p className="mb-4">
          You can use this generated prompt in another AI chat session to ask the AI to implement the suggested changes. This is a "meta-prompt" designed for clarity and effectiveness.
        </p>
        <div className="relative bg-dark-navy rounded-lg border border-light-navy">
          <button
            onClick={handleCopy}
            className="absolute top-3 right-3 p-2 bg-light-navy/50 rounded-md text-slate hover:text-accent hover:bg-light-navy transition-colors duration-200 z-10"
            aria-label="Copy prompt"
          >
            {copied ? <CheckIcon className="h-5 w-5 text-accent" /> : <CopyIcon className="h-5 w-5" />}
          </button>
          <pre className="p-4 pt-12 overflow-auto max-h-[60vh]">
            <code className="text-sm whitespace-pre-wrap font-mono text-lightest-slate">
              {prompt}
            </code>
          </pre>
        </div>
      </div>
    </div>
  );
});