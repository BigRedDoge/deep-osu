// parser/parse_osu.js
// Import necessary classes and functions
import { BeatmapDecoder, HittableObject, SlidableObject, SpinnableObject } from 'osu-parsers';
// Import enums and utility classes from osu-classes
import { HitType, PathType } from 'osu-classes';
import { resolve } from 'path';

// --- Argument Handling ---
const args = process.argv.slice(2);
if (args.length !== 1) {
    console.error('Usage: node parse_osu.js <path_to_osu_file>'); // To stderr
    process.exit(1);
}
const osuFilePath = resolve(args[0]);

// --- Main Parsing Function ---
async function parseAndOutputBeatmap() {
    // Redirect console.log temporarily IF NEEDED - usually bad practice, but for debugging...
    // const originalConsoleLog = console.log;
    // console.log = console.error; // Send any accidental logs to stderr

    const decoder = new BeatmapDecoder();
    let outputData = null; // Initialize outputData

    try {
        const beatmap = await decoder.decodeFromPath(osuFilePath, {
             parseStoryboard: false,
             parseMetadata: true
            });

        // --- Construct the outputData object (same as before) ---
        outputData = {
            fileFormat: beatmap.fileFormat,
            metadata: {
                title: beatmap.metadata.title,
                titleUnicode: beatmap.metadata.titleUnicode,
                artist: beatmap.metadata.artist,
                artistUnicode: beatmap.metadata.artistUnicode,
                creator: beatmap.metadata.creator,
                version: beatmap.metadata.version,
                source: beatmap.metadata.source,
                tags: beatmap.metadata.tags,
                beatmapId: beatmap.metadata.beatmapId,
                beatmapSetId: beatmap.metadata.beatmapSetId,
            },
            hitObjects: beatmap.hitObjects.map(ho => {
                const baseData = { /* ... same mapping logic ... */
                    startTime: ho.startTime,
                    hitType: ho.hitType,
                    objectType: 'Unknown',
                    startX: ho.startPosition.x,
                    startY: ho.startPosition.y,
                    isNewCombo: false,
                };
                if (ho instanceof HittableObject) { baseData.objectType = 'Circle'; baseData.isNewCombo = ho.isNewCombo; }
                else if (ho instanceof SlidableObject) {
                    baseData.objectType = 'Slider'; baseData.isNewCombo = ho.isNewCombo; baseData.repeats = ho.repeats;
                    const curveType = PathType[ho.path.curveType]; let curveTypeChar = 'B';
                    switch(curveType) { case 'Linear': curveTypeChar = 'L'; break; case 'PerfectCurve': curveTypeChar = 'P'; break; case 'Catmull': curveTypeChar = 'C'; break; case 'Bezier': curveTypeChar = 'B'; break; }
                    baseData.path = { curveTypeChar: curveTypeChar, controlPoints: ho.path.controlPoints.map(cp => ({ x: cp.position.x, y: cp.position.y })), expectedDistance: ho.path.expectedDistance };
                }
                else if (ho instanceof SpinnableObject) { baseData.objectType = 'Spinner'; baseData.isNewCombo = ho.isNewCombo; baseData.endTime = ho.endTime; }
                return baseData;
             })
        };
        // --- End constructing outputData ---

        // --- Output JSON to stdout - ONLY THIS SHOULD PRINT TO STDOUT ON SUCCESS ---
        // Ensure no other console logs interfere before this point.
        const finalJsonString = JSON.stringify(outputData, null, 0); // Compact JSON output
        process.stdout.write(finalJsonString);

    } catch (err) {
        // --- Output Errors ONLY to stderr ---
        console.error(`Error parsing beatmap (${osuFilePath}): ${err.message}`);
        console.error(JSON.stringify({ error: `Node parser failed: ${err.message}` })); // Structured error
        process.exit(1); // Exit with non-zero code
    } finally {
        // Restore console.log if you redirected it
        // console.log = originalConsoleLog;
    }
}

// --- Execute ---
parseAndOutputBeatmap();
