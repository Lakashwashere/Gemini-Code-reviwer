# Gemini Code Reviewer (Fresh Start)

This is a web application that uses the Google Gemini API to review code. This new version has been rebuilt from scratch to ensure stability and provide a clean foundation for future features.

## Running Locally

**Prerequisites:** Node.js

1.  **Install dependencies:**
    `npm install`

2.  **Create a `.env` file** in the root directory. You will need to add your Gemini API key and a GitHub Personal Access Token.
    ```
    # Your Google Gemini API Key
    VITE_GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

    # Your GitHub Personal Access Token (with 'public_repo' scope) for fetching repositories
    VITE_GITHUB_PAT="YOUR_GITHUB_PAT"
    ```

3.  **Run the development server:**
    `npm run dev`

4. Open your browser to the URL provided by the development server.