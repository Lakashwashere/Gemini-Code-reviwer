import React, { useRef, useState } from 'react';
import Editor from 'react-simple-code-editor';
import Prism from 'prismjs';
import { PROGRAMMING_LANGUAGES, LANGUAGE_EXTENSIONS, IGNORED_FILES_AND_DIRS } from '../constants';
import { fetchRepoContents } from '../services/githubService';
import { Loader } from './Loader';
import { SparklesIcon } from './icons/SparklesIcon';
import { UploadIcon } from './icons/UploadIcon';
import { FolderIcon } from './icons/FolderIcon';
import { GitHubIcon } from './icons/GitHubIcon';

interface CodeInputProps {
  code: string;
  setCode: (code: string) => void;
  language: string;
  setLanguage: (language: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  onCancel: () => void;
}

const getDefaultProjectLanguage = (currentLanguage: string) => {
    return PROGRAMMING_LANGUAGES.includes('TypeScript')
        ? 'TypeScript'
        : PROGRAMMING_LANGUAGES.includes('JavaScript')
          ? 'JavaScript'
          : currentLanguage;
};

export const CodeInput: React.FC<CodeInputProps> = ({ code, setCode, language, setLanguage, onSubmit, isLoading, onCancel }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [githubUrl, setGithubUrl] = useState('');
  const [isFetchingRepo, setIsFetchingRepo] = useState(false);
  const [githubError, setGithubError] = useState<string | null>(null);

  const handleFetchFromGithub = async () => {
    if (!githubUrl.trim()) {
      setGithubError("Please enter a GitHub URL.");
      return;
    }
    setIsFetchingRepo(true);
    setGithubError(null);
    setCode('');
    try {
      const repoCode = await fetchRepoContents(githubUrl);
      if(!repoCode) {
        throw new Error("Could not fetch any reviewable files from the repository.");
      }
      setCode(repoCode);
      setLanguage(getDefaultProjectLanguage(language));
      setGithubUrl(''); // Clear input on success
    } catch (err) {
      setGithubError(err instanceof Error ? err.message : "An unknown error occurred.");
    } finally {
      setIsFetchingRepo(false);
    }
  };

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
        const pathParts = file.webkitRelativePath.split('/');
        
        if (!LANGUAGE_EXTENSIONS[extension] || pathParts.some(part => IGNORED_FILES_AND_DIRS.includes(part))) {
          console.log(`Skipping file: ${file.webkitRelativePath}`);
          resolve();
          return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string;
          // Add a space after the colon to match the parser
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
      setLanguage(getDefaultProjectLanguage(language));
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
          className="w-full bg-light-navy border border-light-navy text-lightest-slate rounded-md p-2 focus:ring-2 focus:ring-accent focus:outline-none mb-4 appearance-none custom-select-arrow"
        >
          {PROGRAMMING_LANGUAGES.map((lang) => (
            <option key={lang} value={lang} className="bg-light-navy text-lightest-slate">
              {lang}
            </option>
          ))}
        </select>
        
        <div className="mb-4">
          <label htmlFor="github-url" className="block text-sm font-medium text-slate mb-2">
            Import from GitHub
          </label>
          <div className="flex space-x-2">
              <div className="relative flex-grow">
                <GitHubIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate" />
                <input
                  id="github-url"
                  type="url"
                  value={githubUrl}
                  onChange={(e) => setGithubUrl(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleFetchFromGithub()}
                  placeholder="https://github.com/owner/repo"
                  className="w-full bg-light-navy border border-light-navy text-lightest-slate rounded-md py-2 pr-3 pl-10 focus:ring-2 focus:ring-accent focus:outline-none"
                  disabled={isFetchingRepo}
                  aria-label="GitHub repository URL"
                />
              </div>
              <button
                  onClick={handleFetchFromGithub}
                  disabled={isFetchingRepo || !githubUrl.trim()}
                  className="flex items-center justify-center bg-light-navy border border-light-navy text-lightest-slate font-medium py-2 px-4 rounded-md transition-colors duration-200 hover:bg-light-navy/70 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-navy focus:ring-accent disabled:opacity-50 disabled:cursor-not-allowed"
              >
                  {isFetchingRepo ? <Loader className="h-5 w-5" /> : 'Fetch'}
              </button>
          </div>
          {githubError && <p className="text-red-400 text-sm mt-2">{githubError}</p>}
        </div>

        <div className="flex justify-between items-center mb-2">
            <label htmlFor="code-input" className="block text-sm font-medium text-slate">
                Or Paste / Upload
            </label>
            <div className="flex items-center space-x-4">
              <input
                type="file"
                ref={folderInputRef}
                onChange={handleFolderChange}
                className="hidden"
                // @ts-ignore - `webkitdirectory` is a non-standard attribute for folder selection.
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
      <div className="mt-6">
        <button
          onClick={onSubmit}
          disabled={isLoading || !code.trim()}
          className="w-full flex items-center justify-center border border-accent text-accent font-mono py-3 px-4 rounded-md transition-all duration-300 hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-navy focus:ring-accent disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent"
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
        {isLoading && (
            <button
                onClick={onCancel}
                className="w-full mt-2 flex items-center justify-center border border-slate text-slate font-mono py-3 px-4 rounded-md transition-all duration-300 hover:bg-slate/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-navy focus:ring-slate"
            >
                Cancel
            </button>
        )}
      </div>
    </div>
  );
};