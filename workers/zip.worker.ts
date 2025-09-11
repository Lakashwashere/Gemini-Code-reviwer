import JSZip from 'jszip';
import { shouldIncludeFile } from '../utils/fileFilter.ts';

// Ensure the worker context is typed correctly.
const ctx: Worker = self as any;

ctx.onmessage = async (event: MessageEvent<File>) => {
    const file = event.data;

    try {
        const zip = await JSZip.loadAsync(file);
        let projectCode = '';
        const filePromises: Promise<void>[] = [];

        zip.forEach((relativePath, zipEntry) => {
            if (zipEntry.dir || !shouldIncludeFile(relativePath)) {
                return;
            }

            const promise = zipEntry.async('string').then(content => {
                projectCode += `// FILE: ${relativePath}\n${content}\n\n`;
            });
            filePromises.push(promise);
        });

        await Promise.all(filePromises);

        if (!projectCode.trim()) {
            throw new Error("No reviewable source code files found in the zip archive.");
        }

        ctx.postMessage({ success: true, code: projectCode });
    } catch (err) {
        ctx.postMessage({ success: false, error: err instanceof Error ? err.message : "Failed to process the zip file." });
    }
};