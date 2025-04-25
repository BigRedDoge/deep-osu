// encoder/encode_osu.js
import {
    BeatmapEncoder,
    HittableObject,
    SlidableObject,
    SpinnableObject
} from 'osu-parsers'; // BeatmapEncoder comes from osu-parsers

// Import core data structures and enums from osu-classes
import {
    Beatmap, BeatmapMetadataSection, BeatmapDifficultySection, BeatmapGeneralSection,
    ControlPointInfo, TimingPoint, DifficultyPoint, SamplePoint, EffectPoint, SampleSet, HitSound,
    PathPoint, Vector2, PathType, HitType, ControlPointType, HitSample // Added ControlPointType for clarity
} from 'osu-classes';

import { readFileSync } from 'fs';

// --- Helper Functions ---
function getPathTypeEnum(char) { /* ... same as before ... */
    switch (char) { case 'L': return PathType.Linear; case 'P': return PathType.PerfectCurve; case 'B': return PathType.Bezier; case 'C': return PathType.Catmull; default: console.warn(`Warning: Unknown slider path type char '${char}'. Defaulting to Bezier.`); return PathType.Bezier; }
}

// --- Main Encoding Logic ---
async function encodeBeatmapFromInput() {
    let inputJson = '';
    let inputData = null;

    try {
        console.warn("DEBUG: Reading JSON from stdin..."); // Log to stderr
        inputJson = readFileSync(0, 'utf-8');
        if (!inputJson) throw new Error("No input JSON received via stdin.");
        console.warn(`DEBUG: Received JSON string (length: ${inputJson.length})`); // Log to stderr

        inputData = JSON.parse(inputJson);
        if (!inputData || typeof inputData !== 'object') throw new Error("Parsed input is not a valid object.");
        console.warn("DEBUG: JSON parsed successfully."); // Log to stderr

        const beatmap = new Beatmap();
        beatmap.fileFormat = 14;

        // --- Apply Defaults and Overrides (Metadata, Difficulty, General) ---
        beatmap.general = new BeatmapGeneralSection(); beatmap.general.mode = 0; beatmap.general.sampleSet = SampleSet.Normal; beatmap.general.stackLeniency = 0.7;
        beatmap.metadata = new BeatmapMetadataSection(); const meta = inputData.metadata || {}; beatmap.metadata.title = meta.title || "AI Generated Title"; beatmap.metadata.titleUnicode = meta.titleUnicode || beatmap.metadata.title; beatmap.metadata.artist = meta.artist || "AI Artist"; beatmap.metadata.artistUnicode = meta.artistUnicode || beatmap.metadata.artist; beatmap.metadata.creator = meta.creator || "TokenToOsu"; beatmap.metadata.version = meta.version || "Generated"; beatmap.metadata.source = meta.source || ""; beatmap.metadata.tags = Array.isArray(meta.tags) ? meta.tags : (meta.tags ? String(meta.tags).split(' ') : []); beatmap.metadata.beatmapId = meta.beatmapId || -1; beatmap.metadata.beatmapSetId = meta.beatmapSetId || -1;
        beatmap.difficulty = new BeatmapDifficultySection(); const diff = inputData.difficulty || {}; beatmap.difficulty.drainRate = diff.HPDrainRate ?? 5; beatmap.difficulty.circleSize = diff.CircleSize ?? 5; beatmap.difficulty.overallDifficulty = diff.OverallDifficulty ?? 5; beatmap.difficulty.approachRate = diff.ApproachRate ?? beatmap.difficulty.overallDifficulty; beatmap.difficulty.sliderMultiplier = diff.SliderMultiplier ?? 1.4; beatmap.difficulty.sliderTickRate = diff.SliderTickRate ?? 1;
        console.warn("DEBUG: Defaults and overrides applied."); // Log to stderr

        // --- Process Reconstructed Control Points ---
        beatmap.controlPoints = new ControlPointInfo();
        const receivedControlPoints = inputData.control_points;
        if (receivedControlPoints && Array.isArray(receivedControlPoints) && receivedControlPoints.length > 0) {
             console.warn(`DEBUG: Processing ${receivedControlPoints.length} received control points...`);
             receivedControlPoints.sort((a, b) => (a?.time ?? 0) - (b?.time ?? 0)); // Sort safely

             let lastAddedTime = -Infinity;
             let addedTypesAtTime = new Set();

             receivedControlPoints.forEach((cpData, index) => {
                 // *** ADDED NULL/UNDEFINED CHECK FOR cpData ***
                 if (!cpData || typeof cpData !== 'object') {
                      console.warn(`Warning: Skipping invalid control point data at index ${index}:`, cpData);
                      return; // Skip this iteration
                 }
                 // *** END CHECK ***

                 const time = cpData.time ?? 0;
                 if (time !== lastAddedTime) { lastAddedTime = time; addedTypesAtTime.clear(); }

                 try {
                     // Add Difficulty Point (handles SV)
                     if (!addedTypesAtTime.has(ControlPointType.DifficultyPoint)) {
                          const dp = new DifficultyPoint();
                          dp.sliderVelocity = cpData.sliderVelocity ?? 1.0;
                          beatmap.controlPoints.add(dp, time);
                          addedTypesAtTime.add(ControlPointType.DifficultyPoint);
                     }
                     // Add Timing Point only if Uninherited
                     if (cpData.pointType === 'Uninherited' && !addedTypesAtTime.has(ControlPointType.TimingPoint)) {
                          const tp = new TimingPoint();
                          tp.beatLength = (typeof cpData.beatLength === 'number' && cpData.beatLength > 0) ? cpData.beatLength : 500;
                          tp.timeSignature = cpData.timeSignature ?? 4;
                          beatmap.controlPoints.add(tp, time);
                          addedTypesAtTime.add(ControlPointType.TimingPoint);
                     }
                     // Add Sample Point if relevant data exists
                     // Check if cpData exists before accessing properties
                     if (('sampleSet' in cpData || 'volume' in cpData || 'customIndex' in cpData) && !addedTypesAtTime.has(ControlPointType.SamplePoint)) {
                          const sp = new SamplePoint();
                          // Use safe access with default values
                          sp.sampleSet = SampleSet[cpData.sampleSet] ?? SampleSet.None;
                          sp.customSampleIndex = cpData.customIndex ?? 0; // Error was here
                          sp.volume = cpData.volume ?? 100;
                          beatmap.controlPoints.add(sp, time);
                          addedTypesAtTime.add(ControlPointType.SamplePoint);
                     }
                     // Add EffectPoint if needed (e.g., for Kiai if re-added)

                 } catch (cpError) {
                      console.error(`Error processing control point at index ${index} (time ${time}): ${cpError.message}`);
                 }
             });
             console.warn(`DEBUG: Finished processing control points. Total points added: ${beatmap.controlPoints.allPoints.length}`);
        } else {
             console.warn("Warning: No valid 'control_points' array found in input. Adding default timing.");
             // Add default timing points (same as before)
             const time = 0, beatLength = 500, timeSignature = 4, sampleSet = SampleSet.Normal, volume = 70;
             const defaultTimingPoint = new TimingPoint(); defaultTimingPoint.beatLength = beatLength; defaultTimingPoint.timeSignature = timeSignature; const defaultDifficultyPoint = new DifficultyPoint(); const defaultSamplePoint = new SamplePoint(); defaultSamplePoint.sampleSet = sampleSet; defaultSamplePoint.volume = volume; const defaultEffectPoint = new EffectPoint();
             beatmap.controlPoints.add(defaultTimingPoint, time); beatmap.controlPoints.add(defaultDifficultyPoint, time); beatmap.controlPoints.add(defaultSamplePoint, time); beatmap.controlPoints.add(defaultEffectPoint, time);
        }


        // --- Process Hit Objects ---
        const receivedHitObjects = inputData.hit_objects;
        console.log(receivedHitObjects);
        if (receivedHitObjects && Array.isArray(receivedHitObjects)) {
            console.warn(`DEBUG: Processing ${receivedHitObjects.length} received hit objects...`);
            receivedHitObjects.forEach((objData, index) => {
                // *** ADDED NULL/UNDEFINED CHECK FOR objData ***
                 if (!objData || typeof objData !== 'object') {
                      console.warn(`Warning: Skipping invalid hit object data at index ${index}:`, objData);
                      return; // Skip this iteration
                 }
                // *** END CHECK ***

                let hitObject = null;
                const startTime = objData.time ?? 0;
                console.warn(`\nDEBUG: --- Processing Hit Object ${index} at time ${startTime} ---`);

                const startX = objData.x ?? 256; const startY = objData.y ?? 192; const startPos = new Vector2(startX, startY);
                const isNewCombo = objData.is_new_combo ?? false; const hitSound = HitSound.Normal; let currentHitType = 0;
                if (isNewCombo) { currentHitType |= HitType.NewCombo; }

                try {
                    console.warn(`DEBUG: Object Type from Python: ${objData.object_type}`);
                    switch (objData.object_type) {
                        // (Cases remain the same as before)
                        case 'Circle': currentHitType |= HitType.Normal; hitObject = new HittableObject(); hitObject.isNewCombo = isNewCombo; console.warn("DEBUG: HittableObject created."); break;
                        case 'Slider':
                            console.warn("DEBUG: Creating SlidableObject..."); currentHitType |= HitType.Slider; const slider = new SlidableObject(); slider.isNewCombo = isNewCombo; slider.repeats = objData.repeats ?? 0; console.warn(`DEBUG: Slider repeats set to: ${slider.repeats}`);
                            if (objData.points && Array.isArray(objData.points) && objData.points.length >= 2) {
                                const pathType = getPathTypeEnum(objData.slider_type_char || 'B'); const controlPoints = []; controlPoints.push(new PathPoint(new Vector2(0, 0), pathType)); console.warn(`DEBUG: Slider path type: ${PathType[pathType]}, Points received: ${objData.points.length}`);
                                for (let i = 1; i < objData.points.length; ++i) { const p = objData.points[i]; const pointX = p?.x ?? startX; const pointY = p?.y ?? startY; const relativePos = new Vector2(pointX, pointY).subtract(startPos); controlPoints.push(new PathPoint(relativePos)); }
                                slider.path.controlPoints = controlPoints; console.warn(`DEBUG: Slider path created with ${controlPoints.length} points.`);
                            } else { console.warn(`Warning: Slider at time ${startTime} has insufficient points (${objData.points?.length}). Creating minimal path.`); slider.path.controlPoints.push(new PathPoint(new Vector2(0, 0), PathType.Linear)); slider.path.controlPoints.push(new PathPoint(new Vector2(1, 0))); }
                            hitObject = slider; console.warn("DEBUG: SlidableObject created."); break;
                        case 'Spinner': console.warn("DEBUG: Creating SpinnableObject..."); currentHitType |= HitType.Spinner; const spinner = new SpinnableObject(); spinner.isNewCombo = isNewCombo; spinner.endTime = objData.end_time ?? (startTime + 1000); hitObject = spinner; console.warn(`DEBUG: SpinnableObject created, end time: ${spinner.endTime}`); break;
                        default: console.warn(`Warning: Unknown hit object type '${objData.object_type}'. Skipping.`); return;
                    }

                    // Check if hitObject was successfully created before proceeding
                    if (!hitObject) {
                         console.error(`Error: Failed to create hit object instance for type ${objData.object_type}`);
                         return; // Skip to next object
                    }

                    console.warn("DEBUG: Assigning common properties...");
                    hitObject.startTime = startTime; hitObject.startPosition = startPos; hitObject.hitType = currentHitType; hitObject.hitSound = hitSound;
                    hitObject.samples = [new HitSample()]; // Create default samples
                    hitObject.samples[0].sampleSet = SampleSet.Normal;
                    hitObject.samples[0].hitSound = HitSound.Normal;
                    console.log(hitObject);
                    console.warn(`DEBUG: Properties assigned: time=${hitObject.startTime}, type=${hitObject.hitType}`);

                    console.warn(`DEBUG: Applying defaults for object at time ${startTime}...`);
                    hitObject.applyDefaults(beatmap.controlPoints, beatmap.difficulty);
                    console.warn(`DEBUG: Defaults applied for object at time ${startTime}. Slider duration (if applicable): ${hitObject.duration ?? 'N/A'}`);

                    console.warn(`DEBUG: Pushing object type ${objData.object_type} to beatmap.hitObjects`);
                    beatmap.hitObjects.push(hitObject);
                    console.warn(`DEBUG: Object pushed. Current hitObjects count: ${beatmap.hitObjects.length}`);

                } catch (objError) {
                    console.error(`ERROR processing hit object at index ${index} (time ${startTime}): ${objError.message}`);
                    console.error(objError.stack);
                }
            });
            console.warn(`DEBUG: Finished processing hit objects loop. Final count in beatmap: ${beatmap.hitObjects.length}`);
        } else {
            console.warn("Warning: No 'hit_objects' array found or array is empty in input data.");
        }

        // --- Encode Beatmap to String ---
        const encoder = new BeatmapEncoder();
        console.warn("DEBUG: Encoding beatmap to string...");
        const osuString = encoder.encodeToString(beatmap);
        console.warn("osu string", osuString);
        console.warn("DEBUG: Encoding complete.");
        process.stdout.write(osuString); // Write final result to stdout

    } catch (err) {
        console.error(`Error during Node.js encoding process: ${err.message}`); console.error(err.stack); process.exit(1);
    }
}
encodeBeatmapFromInput();


//hitObject.samples = new HitSample(); // Create default samples
