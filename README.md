# Gemini Code Reviewer

This project is a web application that uses the Google Gemini API to review code from various sources, including direct input, file uploads, and GitHub repositories.

## Running Locally

**Prerequisites:** Node.js

1.  Install dependencies:
    `npm install`
2.  Create a `.env` file in the root directory and add your Gemini API key and a GitHub Personal Access Token. The GitHub token is required to prevent rate-limiting when fetching public repositories and needs the `public_repo` scope.
    ```
    VITE_GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    VITE_GITHUB_PAT="YOUR_GITHUB_PERSONAL_ACCESS_TOKEN"
    ```
3.  Run the development server:
    `npm run dev`
4. Open your browser to the URL provided by the development server.
