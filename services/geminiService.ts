import { GoogleGenAI, Type } from "@google/genai";
import type { ReviewFeedback } from '../types.ts';

// This is a Vite-specific feature to access environment variables.
// The variable is replaced at build time.
declare var process: {
  env: {
    API_KEY: string;
  }
};

const reviewSchema = {
  type: Type.OBJECT,
  properties: {
    summary: {
      type: Type.STRING,
      description: "A concise, high-level summary of the code quality."
    },
    suggestions: {
      type: Type.ARRAY,
      description: "A list of specific, actionable suggestions for improvement.",
      items: {
        type: Type.OBJECT,
        properties: {
          // FIX: Add 'file' property to the suggestion schema to support multi-file reviews.
          file: {
            type: Type.STRING,
            description: "The file path where the suggestion applies. Omit if not applicable to a specific file."
          },
          category: {
            type: Type.STRING,
            description: "Category of the suggestion (e.g., Readability, Performance, Security)."
          },
          description: {
            type: Type.STRING,
            description: "A clear description of the issue."
          },
          suggestion: {
            type: Type.STRING,
            description: "The suggested code fix or improvement."
          }
        },
        required: ["category", "description", "suggestion"],
      }
    },
    // FIX: Add 'explanation' property to the schema to get detailed reasoning from the AI.
    explanation: {
        type: Type.STRING,
        description: "A detailed explanation of the major changes made in the revised code and the reasoning behind them."
    },
    revisedCode: {
        type: Type.STRING,
        description: "The full, revised version of the original code with all suggestions applied."
    },
  },
  // FIX: Add 'explanation' to the list of required properties.
  required: ["summary", "suggestions", "explanation", "revisedCode"],
};

export const getCodeReview = async (code: string): Promise<ReviewFeedback> => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API key is not configured. Please set up your .env file.");
  }
  const ai = new GoogleGenAI({ apiKey });

  const systemInstruction = `
    You are an expert code reviewer AI. Your task is to provide a thorough and constructive review of the user-provided code.
    Analyze for readability, performance, security vulnerabilities, and adherence to best practices.
    For multi-file projects, suggestions must include the 'file' property indicating the relevant file path.
    You must provide a high-level explanation of your changes.
    You must respond in a JSON format that adheres to the provided schema.
  `;

  const userPrompt = `Please review the following code:\n\n\`\`\`\n${code}\n\`\`\``;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: userPrompt,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        responseSchema: reviewSchema,
        temperature: 0.3,
      },
    });

    const jsonText = response.text.trim();
    return JSON.parse(jsonText) as ReviewFeedback;

  } catch (error) {
    console.error("Error fetching code review from Gemini API:", error);
    const errorMessage = error instanceof Error ? error.message : "An unknown error occurred";
    if (errorMessage.includes('API key')) {
        throw new Error("An error occurred with the API key. Please check if it's valid.");
    }
    throw new Error("Failed to get a response from the AI. Please check the console for details.");
  }
};