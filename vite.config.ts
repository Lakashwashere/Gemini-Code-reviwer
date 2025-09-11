import { defineConfig, loadEnv } from 'vite';
import path from 'path';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');
    return {
      define: {
        // Standardize on process.env.API_KEY in the app.
        // Read from VITE_GEMINI_API_KEY (Vite convention) or GEMINI_API_KEY from .env
        'process.env.API_KEY': JSON.stringify(env.VITE_GEMINI_API_KEY || env.GEMINI_API_KEY),
        
        // Standardize on process.env.GITHUB_PAT in the app.
        // Read from VITE_GITHUB_PAT (Vite convention) or GITHUB_PAT from .env
        'process.env.GITHUB_PAT': JSON.stringify(env.VITE_GITHUB_PAT || env.GITHUB_PAT),
      },
      resolve: {
        alias: {
          // FIX: Use process.cwd() which is more reliable than __dirname in some environments.
          '@': path.resolve(process.cwd()),
        }
      },
      worker: {
        format: 'es',
      },
    };
});