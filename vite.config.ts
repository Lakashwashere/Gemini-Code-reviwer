import { defineConfig, loadEnv } from 'vite';
import process from 'process';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');
    return {
      define: {
        'process.env.API_KEY': JSON.stringify(env.VITE_GEMINI_API_KEY),
        'process.env.GITHUB_PAT': JSON.stringify(env.VITE_GITHUB_PAT)
      },
    };
});