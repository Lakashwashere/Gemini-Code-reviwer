import { LANGUAGE_EXTENSIONS, IGNORED_FILES_AND_DIRS } from '../constants';

const GITHUB_API_BASE = 'https://api.github.com';

// Type definitions for GitHub API responses
interface RepoDetails {
  default_branch: string;
}

interface TreeItem {
  path: string;
  type: 'blob' | 'tree';
  url: string;
  size?: number;
}

interface TreeResponse {
  tree: TreeItem[];
  truncated: boolean;
}

const parseRepoUrl = (url: string): { owner: string; repo: string } | null => {
  try {
    const urlObj = new URL(url);
    if (urlObj.hostname !== 'github.com') {
      return null;
    }
    const pathParts = urlObj.pathname.split('/').filter(p => p);
    if (pathParts.length >= 2) {
      return { owner: pathParts[0], repo: pathParts[1].replace(/\.git$/, '') };
    }
    return null;
  } catch (e) {
    return null; // Invalid URL
  }
};

const shouldIncludeFile = (path: string): boolean => {
  const extension = path.split('.').pop()?.toLowerCase() || '';
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

export const fetchRepoContents = async (repoUrl: string): Promise<string> => {
  const repoInfo = parseRepoUrl(repoUrl);
  if (!repoInfo) {
    throw new Error("Invalid GitHub repository URL. Format should be: https://github.com/owner/repo");
  }
  const { owner, repo } = repoInfo;

  const repoDetailsResponse = await fetch(`${GITHUB_API_BASE}/repos/${owner}/${repo}`);
  if (!repoDetailsResponse.ok) {
     if (repoDetailsResponse.status === 404) {
        throw new Error(`Repository not found. Please check the URL.`);
     }
    throw new Error(`Failed to fetch repository details. Status: ${repoDetailsResponse.status}`);
  }
  const repoDetails: RepoDetails = await repoDetailsResponse.json();
  const defaultBranch = repoDetails.default_branch;

  const treeResponse = await fetch(`${GITHUB_API_BASE}/repos/${owner}/${repo}/git/trees/${defaultBranch}?recursive=1`);
  if (!treeResponse.ok) {
    throw new Error(`Failed to fetch repository file tree. Status: ${treeResponse.status}`);
  }
  const treeData: TreeResponse = await treeResponse.json();

  if (treeData.truncated) {
    console.warn("Repository tree is too large and has been truncated. Not all files may be included.");
  }

  const filesToFetch = treeData.tree.filter(item => item.type === 'blob' && shouldIncludeFile(item.path));

  if (filesToFetch.length === 0) {
    throw new Error("No reviewable source code files found in the repository. Check if the repo contains supported file types.");
  }
  
  if (filesToFetch.length > 200) {
    console.warn(`Fetching ${filesToFetch.length} files. This may be slow.`);
  }

  const fileContentPromises = filesToFetch.map(async (file) => {
    try {
      const rawFileUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${defaultBranch}/${file.path}`;
      const rawFileResponse = await fetch(rawFileUrl);

      if (!rawFileResponse.ok) {
        console.error(`Failed to fetch raw file for ${file.path}. Status: ${rawFileResponse.status}`);
        return null;
      }
      const content = await rawFileResponse.text();
      // Add a space after the colon to match the parser
      return `// FILE: ${file.path}\n${content}\n\n`;
    } catch (e) {
      console.error(`Error processing file ${file.path}:`, e);
      return null;
    }
  });

  const fileContents = await Promise.all(fileContentPromises);

  return fileContents.filter(content => content !== null).join('');
};