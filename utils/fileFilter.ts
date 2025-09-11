import { LANGUAGE_EXTENSIONS, IGNORED_FILES_AND_DIRS } from '../constants.ts';

export const shouldIncludeFile = (path: string): boolean => {
  // Ignore files starting with a dot (hidden files) unless they are in the explicit extension list
  const fileName = path.split('/').pop() || '';
  const extension = fileName.split('.').pop()?.toLowerCase() || '';

  if (fileName.startsWith('.') && !LANGUAGE_EXTENSIONS[extension]) {
      return false;
  }
  
  if (!LANGUAGE_EXTENSIONS[extension]) {
    return false;
  }

  const pathParts = path.split('/');
  for (const part of pathParts) {
    if (IGNORED_FILES_AND_DIRS.includes(part)) {
      return false;
    }
  }

  return true;
};