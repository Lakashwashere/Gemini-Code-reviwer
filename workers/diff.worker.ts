import { diffLines, Change } from 'diff';

interface DiffData {
    originalContent: string;
    revisedContent: string;
}

// Ensure the worker context is typed correctly.
const ctx: Worker = self as any;

ctx.onmessage = (event: MessageEvent<DiffData>) => {
    const { originalContent, revisedContent } = event.data;
    const diffResult: Change[] = diffLines(originalContent, revisedContent);
    ctx.postMessage(diffResult);
};
