<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1iMRwapYsF4zahllyooFz76WqOrU2-qB4

## Run Locally

**Prerequisites:**  Node.js

1.  Install dependencies:
    `npm install`
2.  Create a `.env.local` file in the root directory and add your keys:
    ```
    VITE_GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    VITE_GITHUB_PAT=YOUR_GITHUB_PERSONAL_ACCESS_TOKEN
    ```
    *   The `VITE_GITHUB_PAT` is optional but highly recommended to avoid GitHub API rate limits.
3.  Run the app:
    `npm run dev`
