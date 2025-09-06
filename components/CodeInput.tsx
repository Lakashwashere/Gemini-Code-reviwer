import React, { useRef } from 'react';
import Editor from 'react-simple-code-editor';
import Prism from 'prismjs';
import { PROGRAMMING_LANGUAGES, LANGUAGE_EXTENSIONS } from '../constants';
import { Loader } from './Loader';
import { SparklesIcon } from './icons/SparklesIcon';
import { UploadIcon } from './icons/UploadIcon';
import { FolderIcon } from './icons/FolderIcon';

interface CodeInputProps {
  code: string;
  setCode: (code: string) => void;
  language: string;
  setLanguage: (language: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export const CodeInput: React.FC<CodeInputProps> = ({ code, setCode, language, setLanguage, onSubmit, isLoading }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setCode(content);

      const extension = file.name.split('.').pop()?.toLowerCase();
      if (extension && LANGUAGE_EXTENSIONS[extension]) {
        const detectedLanguage = LANGUAGE_EXTENSIONS[extension];
        if (PROGRAMMING_LANGUAGES.includes(detectedLanguage)) {
          setLanguage(detectedLanguage);
        }
      }
    };
    reader.onerror = () => {
      console.error("Failed to read the file.");
    }
    reader.readAsText(file);

    event.target.value = '';
  };

  const handleFolderChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    let projectCode = '';
    const filePromises = Array.from(files).map(file => {
      return new Promise<void>((resolve, reject) => {
        const extension = file.name.split('.').pop()?.toLowerCase() || '';
        if (!LANGUAGE_EXTENSIONS[extension]) {
          console.log(`Skipping unsupported file type: ${file.webkitRelativePath}`);
          resolve();
          return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string;
          projectCode += `// FILE: ${file.webkitRelativePath}\n${content}\n\n`;
          resolve();
        };
        reader.onerror = (e) => {
          console.error(`Failed to read file: ${file.name}`, e);
          reject(e);
        }
        reader.readAsText(file);
      });
    });

    try {
      await Promise.all(filePromises);
      setCode(projectCode);
      // Set a sensible default language hint for projects
      if (PROGRAMMING_LANGUAGES.includes('TypeScript')) {
          setLanguage('TypeScript');
      }
    } catch (err) {
      console.error("Error reading folder contents:", err);
    }

    event.target.value = '';
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };
  
  const handleFolderUploadClick = () => {
    folderInputRef.current?.click();
  };
  
  const highlightCode = (code: string) => {
    const langAlias = language.toLowerCase().replace('++', 'cpp').replace('#', 'csharp');
    const grammar = Prism.languages[langAlias];
    if (grammar) {
      return Prism.highlight(code, grammar, langAlias);
    }
    // Fallback for languages not loaded or aliased
    return code;
  };

  return (
    <div className="bg-navy rounded-lg shadow-lg p-6 flex flex-col max-h-[calc(100vh-8rem)]">
      <div className="flex-1 flex flex-col min-h-0">
        <label htmlFor="language-select" className="block text-sm font-medium text-slate mb-2">
          Language / Project Type
        </label>
        <select
          id="language-select"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="w-full bg-light-navy border border-light-navy text-lightest-slate rounded-md p-2 focus:ring-2 focus:ring-accent focus:outline-none mb-4 appearance-none"
          style={{ backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%2364ffda' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e")`, backgroundPosition: 'right 0.5rem center', backgroundRepeat: 'no-repeat', backgroundSize: '1.5em 1.5em' }}
        >
          {PROGRAMMING_LANGUAGES.map((lang) => (
            <option key={lang} value={lang} className="bg-light-navy text-lightest-slate">
              {lang}
            </option>
          ))}
        </select>
        
        <div className="flex justify-between items-center mb-2">
            <label htmlFor="code-input" className="block text-sm font-medium text-slate">
                Your Code
            </label>
            <div className="flex items-center space-x-4">
              <input
                type="file"
                ref={folderInputRef}
                onChange={handleFolderChange}
                className="hidden"
                // @ts-ignore // webkitdirectory is a non-standard attribute for folder upload
                webkitdirectory=""
                mozdirectory=""
              />
              <button
                  onClick={handleFolderUploadClick}
                  className="flex items-center text-sm font-medium text-slate hover:text-accent transition-colors duration-200 p-1 -mr-1"
                  aria-label="Upload project folder"
              >
                  <FolderIcon className="h-4 w-4 mr-2" />
                  Upload Folder
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept={Object.keys(LANGUAGE_EXTENSIONS).map(ext => `.${ext}`).join(',')}
              />
              <button
                  onClick={handleUploadClick}
                  className="flex items-center text-sm font-medium text-slate hover:text-accent transition-colors duration-200 p-1 -mr-1"
                  aria-label="Upload a code file"
              >
                  <UploadIcon className="h-4 w-4 mr-2" />
                  Upload File
              </button>
            </div>
        </div>
        <div className="flex-1 w-full bg-dark-navy border border-light-navy text-lightest-slate rounded-md font-mono text-sm resize-none focus-within:ring-2 focus-within:ring-accent focus-within:outline-none overflow-auto">
          <Editor
            value={code}
            onValueChange={setCode}
            highlight={highlightCode}
            padding={16}
            placeholder="Paste your code here or upload a file/folder..."
            className="language-clike" // Base styling class for Prism theme
            style={{
                fontFamily: '"Fira Code", "Fira Mono", "Roboto Mono", monospace',
                fontSize: 14,
                minHeight: '100%',
                outline: 'none',
            }}
          />
        </div>
      </div>
      <button
        onClick={onSubmit}
        disabled={isLoading || !code.trim()}
        className="mt-6 w-full flex items-center justify-center border border-accent text-accent font-mono py-3 px-4 rounded-md transition-all duration-300 hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-navy focus:ring-accent disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent"
      >
        {isLoading ? (
          <>
            <Loader className="h-5 w-5 mr-3" />
            Analyzing...
          </>
        ) : (
          <>
            <SparklesIcon className="h-5 w-5 mr-2 text-accent" />
            Review Code
          </>
        )}
      </button>
    </div>
  );
};