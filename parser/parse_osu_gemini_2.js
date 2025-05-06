// parser/parse_osu.js
// Import necessary classes and functions
import { BeatmapDecoder, HittableObject, SlidableObject, SpinnableObject } from 'osu-parsers';
import { HitType, PathType, ControlPointType, TimingPoint, DifficultyPoint, EffectPoint, SamplePoint } from 'osu-classes'; // Import ControlPoint types
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
    const decoder = new BeatmapDecoder();
    let outputData = null;

    try {
        const beatmap = await decoder.decodeFromPath(osuFilePath, {
             parseStoryboard: false,
             parseMetadata: true,
             parseTimingPoints: true // Ensure timing points are parsed
            });

        // --- Construct the outputData object ---
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
            // --- Extract Control Point Data ---
            controlPoints: beatmap.controlPoints.allPoints.map(cp => {
                const cpData = {
                    time: cp.startTime,
                    // Determine type for easier Python processing
                    pointType: 'Unknown', // Default
                    // Common values (might be null/default depending on point type)
                    beatLength: null,
                    sliderVelocity: 1.0, // Default SV multiplier
                    timeSignature: 4, // Default time signature
                    sampleSet: cp.sampleSet?.toString() ?? 'None', // Get string representation
                    customIndex: cp.customSampleIndex ?? 0,
                    volume: cp.volume ?? 100,
                    kiai: false, // Default kiai
                    omitFirstBarLine: false // Default omit bar line
                };

                // Populate specific fields based on the actual point type
                if (cp instanceof TimingPoint) {
                    cpData.pointType = 'Uninherited'; // This defines BPM/BeatLength
                    cpData.beatLength = cp.beatLength;
                    cpData.timeSignature = cp.timeSignature;
                    // Inherit kiai/omit from effect point at same time if exists
                    const effectPoint = beatmap.controlPoints.effectPointAt(cp.startTime);
                    if (effectPoint && effectPoint.startTime === cp.startTime) {
                         cpData.kiai = effectPoint.kiai;
                         cpData.omitFirstBarLine = effectPoint.omitFirstBarLine;
                    }
                     // Inherit sample info from sample point at same time if exists
                    const samplePoint = beatmap.controlPoints.samplePointAt(cp.startTime);
                    if (samplePoint && samplePoint.startTime === cp.startTime) {
                        cpData.sampleSet = samplePoint.sampleSet?.toString() ?? 'None';
                        cpData.customIndex = samplePoint.customSampleIndex ?? 0;
                        cpData.volume = samplePoint.volume ?? 100;
                     }
                } else if (cp instanceof DifficultyPoint) {
                    cpData.pointType = 'Inherited'; // This defines SV multiplier
                    cpData.sliderVelocity = cp.sliderVelocity; // SV Multiplier
                    // Inherit kiai/omit from effect point at same time if exists
                    const effectPoint = beatmap.controlPoints.effectPointAt(cp.startTime);
                    if (effectPoint && effectPoint.startTime === cp.startTime) {
                         cpData.kiai = effectPoint.kiai;
                         cpData.omitFirstBarLine = effectPoint.omitFirstBarLine;
                    }
                    // Inherit sample info from sample point at same time if exists
                    const samplePoint = beatmap.controlPoints.samplePointAt(cp.startTime);
                     if (samplePoint && samplePoint.startTime === cp.startTime) {
                        cpData.sampleSet = samplePoint.sampleSet?.toString() ?? 'None';
                        cpData.customIndex = samplePoint.customSampleIndex ?? 0;
                        cpData.volume = samplePoint.volume ?? 100;
                     }
                } else if (cp instanceof EffectPoint) {
                    // Check if this time already handled by Timing/Difficulty point
                    const timingPoint = beatmap.controlPoints.timingPointAt(cp.startTime);
                    const difficultyPoint = beatmap.controlPoints.difficultyPointAt(cp.startTime);
                    if ((timingPoint && timingPoint.startTime === cp.startTime) || (difficultyPoint && difficultyPoint.startTime === cp.startTime)) {
                        // Already processed, skip standalone effect point
                        return null; // Filter this out later
                    }
                    cpData.pointType = 'EffectOnly'; // Only Kiai/Omit changes
                    cpData.kiai = cp.kiai;
                    cpData.omitFirstBarLine = cp.omitFirstBarLine;
                     // Inherit sample info from sample point at same time if exists
                    const samplePoint = beatmap.controlPoints.samplePointAt(cp.startTime);
                     if (samplePoint && samplePoint.startTime === cp.startTime) {
                        cpData.sampleSet = samplePoint.sampleSet?.toString() ?? 'None';
                        cpData.customIndex = samplePoint.customSampleIndex ?? 0;
                        cpData.volume = samplePoint.volume ?? 100;
                     }
                } else if (cp instanceof SamplePoint) {
                     // Check if this time already handled by Timing/Difficulty/Effect point
                    const timingPoint = beatmap.controlPoints.timingPointAt(cp.startTime);
                    const difficultyPoint = beatmap.controlPoints.difficultyPointAt(cp.startTime);
                    const effectPoint = beatmap.controlPoints.effectPointAt(cp.startTime);
                    if ((timingPoint && timingPoint.startTime === cp.startTime) ||
                        (difficultyPoint && difficultyPoint.startTime === cp.startTime) ||
                        (effectPoint && effectPoint.startTime === cp.startTime))
                    {
                        // Already processed, skip standalone sample point
                        return null; // Filter this out later
                    }
                    cpData.pointType = 'SampleOnly'; // Only Sample changes
                    cpData.sampleSet = cp.sampleSet?.toString() ?? 'None';
                    cpData.customIndex = cp.customSampleIndex ?? 0;
                    cpData.volume = cp.volume ?? 100;
                }

                return cpData;
            }).filter(cpData => cpData !== null), // Filter out null entries
            // --- End Control Point Data ---
            hitObjects: beatmap.hitObjects.map(ho => {
                // --- Hit Object Mapping (same as before) ---
                const baseData = {
                    startTime: ho.startTime,
                    hitType: ho.hitType,
                    objectType: 'Unknown',
                    startX: ho.startPosition.x,
                    startY: ho.startPosition.y,
                    isNewCombo: false,
                };
                if (ho instanceof HittableObject) {
                    baseData.objectType = 'Circle'; 
                    baseData.isNewCombo = ho.isNewCombo; 
                } else if (ho instanceof SlidableObject) {
                    baseData.objectType = 'Slider'; 
                    baseData.isNewCombo = ho.isNewCombo; 
                    baseData.repeats = ho.repeats;
                    /*
                    const curveType = PathType[ho.path.curveType]; 
                    let curveTypeChar = 'B';
                    */
                    const curveType = ho.path.curveType; 
                    let curveTypeChar = 'B';
                    switch(curveType) { 
                        case PathType.Linear: 
                            curveTypeChar = 'L'; 
                            break; 
                        case PathType.PerfectCurve: 
                            curveTypeChar = 'P'; 
                            break; 
                        case PathType.Catmull: 
                            curveTypeChar = 'C';
                            break; 
                        case PathType.Bezier: 
                            curveTypeChar = 'B'; 
                            break; 
                    }
                    baseData.path = { 
                        curveTypeChar: curveTypeChar, 
                        controlPoints: ho.path.controlPoints.map(cp => ({ 
                            x: cp.position.x, y: cp.position.y 
                        })), 
                        expectedDistance: ho.path.expectedDistance 
                    };
                }
                else if (ho instanceof SpinnableObject) { 
                    baseData.objectType = 'Spinner'; 
                    baseData.isNewCombo = ho.isNewCombo; 
                    baseData.endTime = ho.endTime; 
                }
                return baseData;
            })
        };

        //  Output json to stdout 
        process.stdout.write(JSON.stringify(outputData, null, 0));

    } catch (err) {
        //  Output Errors to stderr
        console.error(`Error parsing beatmap (${osuFilePath}): ${err.message}`);
        console.error(JSON.stringify({ error: `Node parser failed: ${err.message}` })); // Structured error
        process.exit(1); // Exit with non-zero code
    }
}

// --- Execute ---
parseAndOutputBeatmap();
