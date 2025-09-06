import React from 'react';

interface ExplanationOutputProps {
  explanation: string;
}

export const ExplanationOutput: React.FC<ExplanationOutputProps> = ({ explanation }) => {
  return (
    <div className="space-y-4">
      <h3 className="text-2xl font-bold text-lightest-slate relative flex items-center gap-4 after:content-[''] after:w-full after:h-[1px] after:bg-light-navy">Explanation</h3>
      <div className="text-light-slate bg-light-navy/50 p-4 rounded-lg border border-light-navy">
        <p className="whitespace-pre-wrap">{explanation}</p>
      </div>
    </div>
  );
};
