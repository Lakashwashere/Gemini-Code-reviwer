import { GoogleGenAI, Type } from "@google/genai";
import type { ReviewFeedback } from '../types';

const reviewSchema = {
  type: Type.OBJECT,
  properties: {
    summary: {
      type: Type.STRING,
      description: "A concise, high-level summary of the code quality, highlighting main strengths and weaknesses."
    },
    suggestions: {
      type: Type.ARRAY,
      description: "A list of specific, actionable suggestions for improvement.",
      items: {
        type: Type.OBJECT,
        properties: {
          file: {
            type: Type.STRING,
            description: "The full path of the file the suggestion applies to. For single file reviews, use a placeholder like 'source.code'."
          },
          category: {
            type: Type.STRING,
            description: "The category of the suggestion (e.g., Readability, Performance, Security, Best Practice, Logic)."
          },
          description: {
            type: Type.STRING,
            description: "A clear and concise description of the issue found in the code."
          },
          suggestion: {
            type: Type.STRING,
            description: "The suggested code fix or improvement. Can include a brief code snippet."
          }
        },
        required: ["file", "category", "description", "suggestion"],
      }
    },
    revisedCode: {
        type: Type.STRING,
        description: "The full, complete, and revised version of the original code with all suggestions applied. This should be a single block of code ready to be copied."
    },
    explanation: {
        type: Type.STRING,
        description: "A detailed, step-by-step explanation of the changes made from the original code to the revised code. Explain *why* each change was made, referencing the suggestions provided."
    }
  },
  required: ["summary", "suggestions", "revisedCode", "explanation"],
};

// FIX: Function signature updated to not accept an API key, as it will be sourced from environment variables.
export const getCodeReview = async (code: string, language: string): Promise<ReviewFeedback> => {
  // FIX: Per coding guidelines, instantiate GoogleGenAI with the API key from process.env.
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  
  const isMultiFile = code.includes('// FILE:');

  const systemInstruction = `
    You are an expert code reviewer AI. Your task is to provide a thorough and constructive review of the user-provided code.
    
    ${isMultiFile 
      ? `The user has provided a multi-file project. Each file's content is preceded by a comment line like '// FILE: path/to/file.ext'. Review the project as a whole, considering inter-file dependencies, project structure, and overall architecture.` 
      : `Analyze the single code file provided.`
    }

    Analyze for:
    - Readability, maintainability, project structure
    - Performance bottlenecks
    - Potential security vulnerabilities
    - Adherence to best practices and conventions for ${language} (and other languages detected in the project)
    - Logic errors or potential bugs
    
    You must respond in a JSON format that adheres to the provided schema. Your response must contain:
    1. A high-level summary of the code quality.
    2. A list of suggestions. Each suggestion MUST include the 'file' property. For single file input, use a placeholder like 'source.code' for the file path.
    3. The complete, revised code. ${isMultiFile ? "The revised code must be returned in the same multi-file format, with '// FILE: ...' markers." : "The revised code must be fully functional."}
    4. A detailed explanation of all changes.
  `;

  const userPrompt = `
    Please review the following ${isMultiFile ? 'project' : language + ' code'}:
    \`\`\`
    ${code}
    \`\`\`
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: userPrompt,
      config: {
        systemInstruction: systemInstruction,
        responseMimeType: "application/json",
        responseSchema: reviewSchema,
        temperature: 0.3,
      },
    });

    const jsonText = response.text.trim();
    const parsedResponse = JSON.parse(jsonText);
    
    return parsedResponse as ReviewFeedback;

  } catch (error) {
    console.error("Error fetching code review from Gemini API:", error);
    const errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
    
    if (errorMessage.includes('JSON')) {
        throw new Error("Failed to get a valid JSON response from the AI. The model may have returned an unexpected format. Please try again.");
    }
    throw new Error("An error occurred while communicating with the AI. This could be a network issue, an invalid API key, or an issue with the AI service. Please check the console for more details.");
  }
};
