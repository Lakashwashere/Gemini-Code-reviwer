import { LANGUAGE_EXTENSIONS } from '../constants.ts';

export type CodeFile = {
  path: string;
  content: string;
};

/**
 * Gets a language identifier for Prism.js from a file path.
 */
export const getPrismLanguageFromPath = (path: string): string => {
  const extension = path.split('.').pop()?.toLowerCase() || '';
  const languageName = LANGUAGE_EXTENSIONS[extension];
  if (languageName) {
    return languageName.toLowerCase().replace('++', 'cpp').replace('#', 'csharp');
  }
  return 'clike'; // default for Prism
};

/**
 * Gets a language identifier for Markdown code blocks from a file path.
 */
export const getMarkdownLanguageFromPath = (path:string): string => {
    const extension = path.split('.').pop()?.toLowerCase() || '';
    const languageName = LANGUAGE_EXTENSIONS[extension];
    if (languageName) {
        const lang = languageName.toLowerCase();
        // Common aliases for markdown
        if (lang === 'c#') return 'csharp';
        if (lang === 'c++') return 'cpp';
        if (lang === 'javascript') return 'js';
        if (lang === 'typescript') return 'ts';
        if (lang === 'bash') return 'sh';
        return lang;
    }
    return ''; // No language hint
};


export const parseMultiFileCode = (code: string): CodeFile[] => {
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