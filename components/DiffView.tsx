import React, { useState, useMemo, useEffect, useRef } from 'react';
import type { Change } from 'diff';
import { parseMultiFileCode } from '../utils/codeParser.ts';
import { FileIcon } from './icons/FileIcon.tsx';
import { Loader } from './Loader.tsx';

interface DiffViewProps {
  originalCode: string;
  revisedCode: string;
}

const DiffLine: React.FC<{ part: Change }> = ({ part }) => {
    const className = part.added ? 'diff-added' : part.removed ? 'diff-removed' : 'diff-common';
    const lines = part.value.replace(/\n$/, '').split('\n');

    return (
        <>
            {lines.map((line, index) => (
                <span key={index} className={`diff-line ${className}`}>
                    {line}
                </span>
            ))}
        </>
    );
};

export const DiffView: React.FC<DiffViewProps> = ({ originalCode, revisedCode }) => {
  const originalFiles = useMemo(() => parseMultiFileCode(originalCode), [originalCode]);
  const revisedFiles = useMemo(() => parseMultiFileCode(revisedCode), [revisedCode]);
  
  const allFilePaths = useMemo(() => {
    const paths = new Set<string>();
    originalFiles.forEach(f => paths.add(f.path));
    revisedFiles.forEach(f => paths.add(f.path));
    return Array.from(paths).sort();
  }, [originalFiles, revisedFiles]);

  const [selectedFilePath, setSelectedFilePath] = useState<string | undefined>(allFilePaths[0]);
  const [diffResult, setDiffResult] = useState<Change[] | null>(null);
  const [isDiffing, setIsDiffing] = useState(false);
  const diffWorkerRef = useRef<Worker | null>(null);

  useEffect(() => {
    // Initialize the diff worker using the standard Web Worker API for browser compatibility.
    diffWorkerRef.current = new Worker(new URL('../workers/diff.worker.ts', import.meta.url), { type: 'module' });
    
    diffWorkerRef.current.onmessage = (event: MessageEvent<Change[]>) => {
      setDiffResult(event.data);
      setIsDiffing(false);
    };

    return () => {
      diffWorkerRef.current?.terminate();
    };
  }, []);

  useEffect(() => {
    if (!selectedFilePath) return;

    setIsDiffing(true);
    setDiffResult(null);

    const originalFile = originalFiles.find(f => f.path === selectedFilePath);
    const revisedFile = revisedFiles.find(f => f.path === selectedFilePath);

    const originalContent = originalFile?.content || '';
    const revisedContent = revisedFile?.content || '';
    
    diffWorkerRef.current?.postMessage({ originalContent, revisedContent });

  }, [selectedFilePath, originalFiles, revisedFiles]);


  const isMultiFile = allFilePaths.length > 1;

  if (allFilePaths.length === 0) {
    return (
      <div className="text-slate bg-light-navy/50 p-4 rounded-lg border border-light-navy">
        Could not parse code to display diff.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-2xl font-bold text-lightest-slate relative flex items-center gap-4 after:content-[''] after:w-full after:h-[1px] after:bg-light-navy">Code Diff</h3>
      <div className={`bg-dark-navy rounded-lg border border-light-navy min-h-[400px] max-h-[70vh] ${isMultiFile ? 'flex' : ''}`}>
        {isMultiFile && (
          <aside className="w-1/3 max-w-xs border-r border-light-navy p-2 overflow-y-auto">
            <ul className="space-y-1">
              {allFilePaths.map((path) => (
                <li key={path}>
                  <button
                    onClick={() => setSelectedFilePath(path)}
                    className={`w-full text-left flex items-center p-2 rounded-md text-sm transition-colors duration-200 ${
                      selectedFilePath === path
                        ? 'bg-light-navy text-lightest-slate'
                        : 'text-slate hover:bg-light-navy/50 hover:text-lightest-slate'
                    }`}
                  >
                    <FileIcon className="h-4 w-4 mr-2 flex-shrink-0" />
                    <span className="truncate" title={path}>{path}</span>
                  </button>
                </li>
              ))}
            </ul>
          </aside>
        )}
        <main className="flex-1 relative">
          <pre className={`p-4 overflow-auto h-full ${isMultiFile ? 'rounded-r-lg' : 'rounded-lg'}`}>
            <code className="diff">
                {isDiffing && (
                  <div className="flex items-center justify-center h-full">
                    <Loader className="h-8 w-8 text-accent" />
                    <span className="ml-4 text-slate">Calculating diff...</span>
                  </div>
                )}
                {!isDiffing && diffResult ? diffResult.map((part, index) => (
                    <DiffLine key={index} part={part} />
                )) : !isDiffing && 'Select a file to view the diff.'}
            </code>
          </pre>
        </main>
      </div>
    </div>
  );
};