
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { parseMultiFileCode } from '../utils/codeParser';
import { PlayIcon } from './icons/PlayIcon';
import { TrashIcon } from './icons/TrashIcon';

interface CodeRunnerProps {
  code: string;
  language: string;
}

interface OutputLine {
  type: 'log' | 'error' | 'warn' | 'info' | 'debug' | 'system';
  timestamp: string;
  message: string;
}

const getLineStyle = (type: OutputLine['type']) => {
  switch (type) {
    case 'error': return 'text-red-400';
    case 'warn': return 'text-yellow-400';
    case 'system': return 'text-accent';
    case 'info': return 'text-blue-400';
    default: return 'text-lightest-slate';
  }
};

export const CodeRunner: React.FC<CodeRunnerProps> = ({ code, language }) => {
  const [outputLines, setOutputLines] = useState<OutputLine[]>([]);
  const iframeContainerRef = useRef<HTMLDivElement>(null);
  const consoleEndRef = useRef<HTMLDivElement>(null);
  
  const files = parseMultiFileCode(code);
  const isRunnable = files.length === 1;
  const isJavaScript = language === 'JavaScript';

  const addOutputLine = useCallback((type: OutputLine['type'], message: string) => {
    const timestamp = new Date().toLocaleTimeString([], { hour12: false });
    setOutputLines(prev => [...prev, { type, timestamp, message }]);
  }, []);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.source !== iframeContainerRef.current?.querySelector('iframe')?.contentWindow) {
        return;
      }
      
      const { source, type, message } = event.data;
      if (source === 'code-runner-iframe') {
        addOutputLine(type, message);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, [addOutputLine]);

  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [outputLines]);

  const handleRun = () => {
    if (!isRunnable || !isJavaScript || !iframeContainerRef.current) return;
    
    iframeContainerRef.current.innerHTML = '';
    setOutputLines([]);

    addOutputLine('system', 'Starting execution...');

    const singleFileContent = files[0].content;
    
    const runnerHtml = `
      <html>
        <head><style>body { margin: 0; }</style></head>
        <body>
          <script type="module">
            const post = (type, message) => window.parent.postMessage({ source: 'code-runner-iframe', type, message }, '*');
            const formatArg = (arg) => {
              if (arg instanceof Error) return arg.stack || arg.message;
              if (typeof arg === 'object' && arg !== null) {
                try { return JSON.stringify(arg, null, 2); } catch (e) { return '[Unserializable Object]'; }
              }
              return String(arg);
            };
            const originalConsole = { ...window.console };
            Object.keys(originalConsole).forEach(key => {
              if (typeof originalConsole[key] === 'function') {
                window.console[key] = (...args) => {
                  post(key, args.map(formatArg).join(' '));
                  originalConsole[key](...args);
                }
              }
            });
            window.addEventListener('error', e => {
              post('error', e.message);
              if(e.error && e.error.stack) post('error', e.error.stack);
            });
            window.addEventListener('unhandledrejection', e => post('error', 'Unhandled promise rejection: ' + (e.reason.stack || e.reason)));
            try {
              ${singleFileContent}
              post('system', 'Execution finished.');
            } catch (e) {
              post('error', e.stack || e.message);
              post('system', 'Execution finished with an error.');
            }
          </script>
        </body>
      </html>
    `;
    
    const iframe = document.createElement('iframe');
    iframe.setAttribute('sandbox', 'allow-scripts');
    iframe.setAttribute('style', 'display: none;');
    iframe.srcdoc = runnerHtml;

    iframeContainerRef.current.appendChild(iframe);
  };

  const handleClear = () => {
    setOutputLines([]);
  };

  return (
    <div className="space-y-4 h-full flex flex-col">
      <h3 className="text-2xl font-bold text-lightest-slate relative flex items-center gap-4 after:content-[''] after:w-full after:h-[1px] after:bg-light-navy">
        Run Code
      </h3>

      <div className="flex items-center space-x-4">
        <button
          onClick={handleRun}
          disabled={!isRunnable || !isJavaScript}
          className="flex items-center justify-center bg-light-navy border border-light-navy text-lightest-slate font-medium py-2 px-4 rounded-md transition-colors duration-200 hover:bg-light-navy/70 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-navy focus:ring-accent disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Run code"
        >
          <PlayIcon className="h-5 w-5 mr-2" />
          Run
        </button>
        <button
          onClick={handleClear}
          className="flex items-center justify-center bg-light-navy border border-light-navy text-lightest-slate font-medium py-2 px-4 rounded-md transition-colors duration-200 hover:bg-light-navy/70 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-navy focus:ring-accent"
          aria-label="Clear console"
        >
          <TrashIcon className="h-5 w-5 mr-2" />
          Clear Console
        </button>
      </div>

      {(!isRunnable || !isJavaScript) && (
        <div className="text-slate bg-light-navy/50 p-4 rounded-lg border border-light-navy">
          <p className="font-semibold text-light-slate">Execution Not Available</p>
          <p className="text-sm">
            The code runner currently only supports single-file JavaScript projects.
            {!isRunnable && ` This project contains multiple files.`}
            {!isJavaScript && ` The selected language is ${language}, not JavaScript.`}
          </p>
        </div>
      )}

      <div className="flex-1 min-h-0 bg-dark-navy rounded-lg border border-light-navy p-4 overflow-y-auto font-mono text-sm">
        {outputLines.map((line, index) => (
          <div key={index} className={`flex items-start whitespace-pre-wrap ${getLineStyle(line.type)}`}>
            <span className="text-slate/60 mr-4">{line.timestamp}</span>
            <span className="flex-1">{line.message}</span>
          </div>
        ))}
        <div ref={consoleEndRef} />
      </div>

      <div ref={iframeContainerRef} style={{ display: 'none' }} />
    </div>
  );
};