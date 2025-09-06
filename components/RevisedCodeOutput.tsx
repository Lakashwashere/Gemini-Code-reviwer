import React, { useState, useEffect, useMemo, useRef } from 'react';
import Prism from 'prismjs';
import { CopyIcon } from './icons/CopyIcon';
import { CheckIcon } from './icons/CheckIcon';
import { FileIcon } from './icons/FileIcon';
import { LANGUAGE_EXTENSIONS } from '../constants';

type CodeFile = {
  path: string;
  content: string;
};

const getLanguageFromPath = (path: string): string => {
  const extension = path.split('.').pop()?.toLowerCase() || '';
  const languageName = LANGUAGE_EXTENSIONS[extension];
  if (languageName) {
    return languageName.toLowerCase().replace('++', 'cpp').replace('#', 'csharp');
  }
  return 'clike'; // default
};

const parseMultiFileCode = (code: string): CodeFile[] => {
  if (!code.includes('// FILE:')) {
    return [{ path: 'source.code', content: code.trim() }];
  }
  
  const files = code.split('// FILE: ').slice(1).map(part => {
    const newlineIndex = part.indexOf('\n');
    const path = part.substring(0, newlineIndex).trim();
    const content = part.substring(newlineIndex + 1).trim();
    return { path, content };
  });
  
  return files.filter(f => f.path && f.content);
};


interface RevisedCodeOutputProps {
  code: string;
}

export const RevisedCodeOutput: React.FC<RevisedCodeOutputProps> = ({ code }) => {
  const [copied, setCopied] = useState(false);
  const files = useMemo(() => parseMultiFileCode(code), [code]);
  const [selectedFile, setSelectedFile] = useState<CodeFile | undefined>(files[0]);
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    setSelectedFile(files[0]);
  }, [files]);

  useEffect(() => {
    if (selectedFile && codeRef.current) {
      Prism.highlightElement(codeRef.current);
    }
  }, [selectedFile]);

  const handleCopy = () => {
    if (!selectedFile) return;
    navigator.clipboard.writeText(selectedFile.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(err => {
        console.error('Failed to copy code: ', err);
    });
  };

  if (!selectedFile) {
    return (
      <div className="text-slate bg-light-navy/50 p-4 rounded-lg border border-light-navy">
        Could not parse or display revised code.
      </div>
    );
  }

  const isMultiFile = files.length > 1;
  const languageClass = `language-${getLanguageFromPath(selectedFile.path)}`;

  return (
    <div className="space-y-4">
       <h3 className="text-2xl font-bold text-lightest-slate relative flex items-center gap-4 after:content-[''] after:w-full after:h-[1px] after:bg-light-navy">Revised Code</h3>
      <div className={`bg-dark-navy rounded-lg border border-light-navy min-h-[400px] max-h-[70vh] ${isMultiFile ? 'flex' : ''}`}>
        {isMultiFile && (
          <aside className="w-1/3 max-w-xs border-r border-light-navy p-2 overflow-y-auto">
            <ul className="space-y-1">
              {files.map((file) => (
                <li key={file.path}>
                  <button
                    onClick={() => setSelectedFile(file)}
                    className={`w-full text-left flex items-center p-2 rounded-md text-sm transition-colors duration-200 ${
                      selectedFile.path === file.path
                        ? 'bg-light-navy text-lightest-slate'
                        : 'text-slate hover:bg-light-navy/50 hover:text-lightest-slate'
                    }`}
                  >
                    <FileIcon className="h-4 w-4 mr-2 flex-shrink-0" />
                    <span className="truncate" title={file.path}>{file.path}</span>
                  </button>
                </li>
              ))}
            </ul>
          </aside>
        )}
        <main className="flex-1 relative">
          <button 
            onClick={handleCopy}
            className="absolute top-3 right-3 p-2 bg-light-navy/50 rounded-md text-slate hover:text-accent hover:bg-light-navy transition-colors duration-200 z-10"
            aria-label="Copy code"
          >
            {copied ? <CheckIcon className="h-5 w-5 text-accent" /> : <CopyIcon className="h-5 w-5" />}
          </button>
          <pre className={`p-4 pt-12 overflow-auto h-full ${isMultiFile ? 'rounded-r-lg' : 'rounded-lg'}`}>
            <code ref={codeRef} className={languageClass}>
              {selectedFile.content}
            </code>
          </pre>
        </main>
      </div>
    </div>
  );
};