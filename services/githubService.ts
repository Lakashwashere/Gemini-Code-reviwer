
import { shouldIncludeFile } from '../utils/fileFilter';

// FIX: Add type declaration for `process` to satisfy TypeScript for `process.env.GITHUB_PAT`.
// Vite will replace `process.env.GITHUB_PAT` with its value at build time.
declare var process: {
  env: {
    GITHUB_PAT: string;
  }
};

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

const handleAuthError = (status: number) => {
    if (status === 401) {
        const authError = `GitHub API authentication failed (401 Unauthorized).`;
        const instruction = `This means the Personal Access Token is invalid, expired, or lacks the correct permissions. Please verify your PAT and ensure it has 'public_repo' scope.`;
        throw new Error(`${authError} ${instruction}`);
    }
    if (status === 403) {
        const rateLimitError = `GitHub API request forbidden (403 Forbidden).`;
        const instruction = `This is likely due to rate limiting. Please ensure your PAT is valid and has 'public_repo' scope.`;
        throw new Error(`${rateLimitError} ${instruction}`);
    }
};

export const fetchRepoContents = async (repoUrl: string): Promise<string> => {
  // FIX: Use `process.env.GITHUB_PAT` directly. Vite replaces this at build time.
  // The `globalThis.process` syntax is incorrect for Vite's `define` replacement and causes type errors.
  const GITHUB_PAT = process.env.GITHUB_PAT;

  // The PAT is now required to prevent rate-limiting on unauthenticated requests.
  if (!GITHUB_PAT) {
    throw new Error(
      'GitHub Personal Access Token is not configured in the execution environment. ' +
      'Please create a token with `public_repo` scope and set it up to fetch repositories.'
    );
  }

  const repoInfo = parseRepoUrl(repoUrl);
  if (!repoInfo) {
    throw new Error("Invalid GitHub repository URL. Format should be: https://github.com/owner/repo");
  }
  const { owner, repo } = repoInfo;

  const headers: HeadersInit = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': `Bearer ${GITHUB_PAT}`,
  };
  
  const repoDetailsResponse = await fetch(`${GITHUB_API_BASE}/repos/${owner}/${repo}`, { headers });
  if (!repoDetailsResponse.ok) {
     if (repoDetailsResponse.status === 404) {
        throw new Error(`Repository not found. Please check the URL.`);
     }
     if (repoDetailsResponse.status === 401 || repoDetailsResponse.status === 403) {
        handleAuthError(repoDetailsResponse.status);
     }
    throw new Error(`Failed to fetch repository details. Status: ${repoDetailsResponse.status}`);
  }
  const repoDetails: RepoDetails = await repoDetailsResponse.json();
  const defaultBranch = repoDetails.default_branch;

  const treeResponse = await fetch(`${GITHUB_API_BASE}/repos/${owner}/${repo}/git/trees/${defaultBranch}?recursive=1`, { headers });
  if (!treeResponse.ok) {
    if (treeResponse.status === 401 || treeResponse.status === 403) {
      handleAuthError(treeResponse.status);
    }
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
      const rawFileResponse = await fetch(rawFileUrl, { headers }); // Pass headers to raw fetch as well for private repos

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