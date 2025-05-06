// --- Imports ---
import { BeatmapEncoder, HittableObject, SlidableObject, SpinnableObject } from 'osu-parsers';
import { Beatmap, BeatmapMetadataSection, BeatmapDifficultySection, BeatmapGeneralSection, ControlPointInfo, TimingPoint, DifficultyPoint, SamplePoint, EffectPoint, SampleSet, HitSound, PathPoint, Vector2, PathType, HitType, ControlPointType, HitSample // Added HitSample
 } from 'osu-classes';
import { readFileSync } from 'fs';

// --- Helper Functions ---
function getPathTypeEnum(char) {
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
        // console.warn(`DEBUG: Received JSON string (length: ${inputJson.length})`); // Log to stderr

        inputData = JSON.parse(inputJson);
        if (!inputData || typeof inputData !== 'object') throw new Error("Parsed input is not a valid object.");
        console.warn("DEBUG: JSON parsed successfully."); // Log to stderr

        const beatmap = new Beatmap();
        beatmap.fileFormat = 14;

        // --- Apply Defaults and Overrides (Metadata, Difficulty, General) ---
        // (Same as before)
        beatmap.general = new BeatmapGeneralSection(); beatmap.general.mode = 0; beatmap.general.sampleSet = SampleSet.Normal; beatmap.general.stackLeniency = 0.7;
        beatmap.metadata = new BeatmapMetadataSection(); const meta = inputData.metadata || {}; beatmap.metadata.title = meta.title || "AI Generated Title"; beatmap.metadata.titleUnicode = meta.titleUnicode || beatmap.metadata.title; beatmap.metadata.artist = meta.artist || "AI Artist"; beatmap.metadata.artistUnicode = meta.artistUnicode || beatmap.metadata.artist; beatmap.metadata.creator = meta.creator || "TokenToOsu"; beatmap.metadata.version = meta.version || "Generated"; beatmap.metadata.source = meta.source || ""; beatmap.metadata.tags = Array.isArray(meta.tags) ? meta.tags : (meta.tags ? String(meta.tags).split(' ') : []); beatmap.metadata.beatmapId = meta.beatmapId || -1; beatmap.metadata.beatmapSetId = meta.beatmapSetId || -1;
        beatmap.difficulty = new BeatmapDifficultySection(); const diff = inputData.difficulty || {}; beatmap.difficulty.drainRate = diff.HPDrainRate ?? 5; beatmap.difficulty.circleSize = diff.CircleSize ?? 5; beatmap.difficulty.overallDifficulty = diff.OverallDifficulty ?? 5; beatmap.difficulty.approachRate = diff.ApproachRate ?? beatmap.difficulty.overallDifficulty; beatmap.difficulty.sliderMultiplier = diff.SliderMultiplier ?? 1.4; beatmap.difficulty.sliderTickRate = diff.SliderTickRate ?? 1;
        console.warn("DEBUG: Defaults and overrides applied."); // Log to stderr

        // --- Process Reconstructed Control Points ---
        // (Same as before - ensure this adds points correctly)
        beatmap.controlPoints = new ControlPointInfo();
        const receivedControlPoints = inputData.control_points;
        if (receivedControlPoints && Array.isArray(receivedControlPoints) && receivedControlPoints.length > 0) {
             console.warn(`DEBUG: Processing ${receivedControlPoints.length} received control points...`);
             receivedControlPoints.sort((a, b) => (a?.time ?? 0) - (b?.time ?? 0));
             let lastAddedTime = -Infinity; let addedTypesAtTime = new Set();
             receivedControlPoints.forEach((cpData, index) => {
                 if (!cpData || typeof cpData !== 'object') { console.warn(`Warning: Skipping invalid control point data at index ${index}:`, cpData); return; }
                 const time = cpData.time ?? 0; if (time !== lastAddedTime) { lastAddedTime = time; addedTypesAtTime.clear(); }
                 try {
                     if (!addedTypesAtTime.has(ControlPointType.DifficultyPoint)) { const dp = new DifficultyPoint(); dp.sliderVelocity = cpData.sliderVelocity ?? 1.0; beatmap.controlPoints.add(dp, time); addedTypesAtTime.add(ControlPointType.DifficultyPoint); }
                     if (cpData.pointType === 'Uninherited' && !addedTypesAtTime.has(ControlPointType.TimingPoint)) { const tp = new TimingPoint(); tp.beatLength = (typeof cpData.beatLength === 'number' && cpData.beatLength > 0) ? cpData.beatLength : 500; tp.timeSignature = cpData.timeSignature ?? 4; beatmap.controlPoints.add(tp, time); addedTypesAtTime.add(ControlPointType.TimingPoint); }
                     if (('sampleSet' in cpData || 'volume' in cpData || 'customIndex' in cpData) && !addedTypesAtTime.has(ControlPointType.SamplePoint)) { const sp = new SamplePoint(); sp.sampleSet = SampleSet[cpData.sampleSet] ?? SampleSet.None; sp.customSampleIndex = cpData.customIndex ?? 0; sp.volume = cpData.volume ?? 100; beatmap.controlPoints.add(sp, time); addedTypesAtTime.add(ControlPointType.SamplePoint); }
                 } catch (cpError) { console.error(`Error processing control point at index ${index} (time ${time}): ${cpError.message}`); }
             });
             console.warn(`DEBUG: Finished processing control points. Total points added: ${beatmap.controlPoints.allPoints.length}`);
        } else {
             console.warn("Warning: No valid 'control_points' array found in input. Adding default timing.");
             const time = 0, beatLength = 500, timeSignature = 4, sampleSet = SampleSet.Normal, volume = 70; const defaultTimingPoint = new TimingPoint(); defaultTimingPoint.beatLength = beatLength; defaultTimingPoint.timeSignature = timeSignature; const defaultDifficultyPoint = new DifficultyPoint(); const defaultSamplePoint = new SamplePoint(); defaultSamplePoint.sampleSet = sampleSet; defaultSamplePoint.volume = volume; const defaultEffectPoint = new EffectPoint(); beatmap.controlPoints.add(defaultTimingPoint, time); beatmap.controlPoints.add(defaultDifficultyPoint, time); beatmap.controlPoints.add(defaultSamplePoint, time); beatmap.controlPoints.add(defaultEffectPoint, time);
        }


        // --- Process Hit Objects ---
        const receivedHitObjects = inputData.hit_objects;
        // console.log(receivedHitObjects); // Keep if needed for deep debug
        if (receivedHitObjects && Array.isArray(receivedHitObjects)) {
            console.warn(`DEBUG: Processing ${receivedHitObjects.length} received hit objects...`);
            receivedHitObjects.forEach((objData, index) => {
                 if (!objData || typeof objData !== 'object') { console.warn(`Warning: Skipping invalid hit object data at index ${index}:`, objData); return; }

                let hitObject = null;
                const startTime = objData.time ?? 0;
                // console.warn(`\nDEBUG: --- Processing Hit Object ${index} at time ${startTime} ---`); // Reduce verbosity

                const startX = objData.x ?? 256; const startY = objData.y ?? 192; const startPos = new Vector2(startX, startY);
                const isNewCombo = objData.is_new_combo ?? false;
                // *** FIX: Read comboOffset from input data ***
                const comboOffset = objData.comboOffset ?? 0; // Default to 0 if missing
                // *** FIX: Determine hitSound based on samples if available, otherwise default ***
                // (We'll assign samples later, but can potentially infer base sound)
                let baseHitSound = HitSound.Normal; // Default base sound

                let currentHitType = 0; // Start with 0
                // Set NewCombo bit based on flag AND combo offset
                if (isNewCombo) {
                     currentHitType |= HitType.NewCombo;
                     // Add combo skip offset bits (bits 4, 5, 6)
                     // Clamp offset to valid range (0-7 for 3 bits)
                     const clampedOffset = Math.max(0, Math.min(comboOffset, 7));
                     currentHitType |= (clampedOffset << 4);
                }

                try {
                    // console.warn(`DEBUG: Object Type from Python: ${objData.object_type}`);
                    switch (objData.object_type) {
                        case 'Circle':
                            currentHitType |= HitType.Normal; // Set object type bit
                            hitObject = new HittableObject();
                            hitObject.isNewCombo = isNewCombo; // Set property for osu-classes internal logic
                            hitObject.comboOffset = comboOffset; // Set property
                            // console.warn("DEBUG: HittableObject created.");
                            break;
                        case 'Slider':
                            currentHitType |= HitType.Slider; // Set object type bit
                            const slider = new SlidableObject();
                            slider.isNewCombo = isNewCombo; // Set property
                            slider.comboOffset = comboOffset; // Set property
                            slider.repeats = objData.repeats ?? 0;
                            // console.warn(`DEBUG: Slider repeats set to: ${slider.repeats}`);
                            if (objData.points && Array.isArray(objData.points) && objData.points.length >= 2) {
                                const pathType = getPathTypeEnum(objData.slider_type_char || 'B');
                                const controlPoints = [];
                                controlPoints.push(new PathPoint(new Vector2(0, 0), pathType));
                                // console.warn(`DEBUG: Slider path type: ${PathType[pathType]}, Points received: ${objData.points.length}`);
                                for (let i = 1; i < objData.points.length; ++i) {
                                    const p = objData.points[i];
                                    const pointX = p?.x ?? startX; const pointY = p?.y ?? startY;
                                    const relativePos = new Vector2(pointX, pointY).subtract(startPos);
                                    controlPoints.push(new PathPoint(relativePos));
                                }
                                slider.path.controlPoints = controlPoints;
                                // *** FIX: Set expectedDistance BEFORE applyDefaults ***
                                if (typeof objData.path?.expectedDistance === 'number') {
                                     slider.path.expectedDistance = objData.path.expectedDistance;
                                     console.warn(`DEBUG: Setting expectedDistance for slider at ${startTime} to ${slider.path.expectedDistance}`);
                                } else {
                                     console.warn(`DEBUG: No expectedDistance provided for slider at ${startTime}. osu-classes will calculate it.`);
                                }
                                // console.warn(`DEBUG: Slider path created with ${controlPoints.length} points.`);
                            } else {
                                console.warn(`Warning: Slider at time ${startTime} has insufficient points (${objData.points?.length}). Creating minimal path.`);
                                slider.path.controlPoints.push(new PathPoint(new Vector2(0, 0), PathType.Linear)); slider.path.controlPoints.push(new PathPoint(new Vector2(1, 0)));
                            }
                            hitObject = slider;
                            // console.warn("DEBUG: SlidableObject created.");
                            break;
                        case 'Spinner':
                            currentHitType |= HitType.Spinner; // Set object type bit
                            const spinner = new SpinnableObject();
                            spinner.isNewCombo = isNewCombo; // Set property
                            spinner.comboOffset = comboOffset; // Set property
                            spinner.endTime = objData.end_time ?? (startTime + 1000);
                            hitObject = spinner;
                            // console.warn(`DEBUG: SpinnableObject created, end time: ${spinner.endTime}`);
                            break;
                        default:
                            console.warn(`Warning: Unknown hit object type '${objData.object_type}'. Skipping.`);
                            return;
                    }

                    if (!hitObject) { console.error(`Error: Failed to create hit object instance for type ${objData.object_type}`); return; }

                    // console.warn("DEBUG: Assigning common properties...");
                    hitObject.startTime = startTime;
                    hitObject.startPosition = startPos;

                    // *** FIX: Reconstruct Samples ***
                    if (objData.samples && Array.isArray(objData.samples) && objData.samples.length > 0) {
                        hitObject.samples = objData.samples.map(sData => {
                             const sample = new HitSample();
                             // Assign properties from sData received from Python
                             sample.hitSound = HitSound[sData.hitSound] ?? HitSound.None; // Map string name to enum
                             sample.sampleSet = SampleSet[sData.sampleSet] ?? SampleSet.None; // Map string name to enum
                             sample.volume = sData.volume ?? 0; // Use provided volume or 0
                             // Construct additions bitmask from sample.hitSound
                             if (sample.hitSound !== HitSound.Normal && sample.hitSound !== HitSound.None) {
                                  baseHitSound |= sample.hitSound; // Add Whistle/Finish/Clap bits to base sound
                             }
                             // Handle custom index/filename if provided by Python
                             // sample.customIndex = sData.customIndex ?? 0;
                             // sample.filename = sData.filename ?? '';
                             return sample;
                        }).filter(s => s.hitSound !== HitSound.None); // Filter out None samples if any

                        // Ensure there's at least a Normal sample if others exist but Normal doesn't
                        if (hitObject.samples.length > 0 && !hitObject.samples.some(s => s.hitSound === HitSound.Normal)) {
                             const normalSample = new HitSample();
                             normalSample.hitSound = HitSound.Normal;
                             // Inherit sample set from first addition or timing point? Let's use timing point default.
                             const sp = beatmap.controlPoints.samplePointAt(startTime);
                             normalSample.sampleSet = sp?.sampleSet ?? SampleSet.Normal;
                             normalSample.volume = sp?.volume ?? 0; // Use timing point volume for base normal sample
                             hitObject.samples.unshift(normalSample); // Add Normal sample at the beginning
                        }
                        // If ONLY Normal sound was intended, baseHitSound remains HitSound.Normal
                        // If additions were present, baseHitSound now includes their bits

                    } else {
                         // Fallback to default sample creation if none provided
                         console.warn(`Warning: No sample data provided for object at ${startTime}. Creating default.`);
                         hitObject.samples = hitObject.createHitSamples(); // Creates default Normal sample
                         baseHitSound = HitSound.Normal; // Ensure base sound is Normal
                    }
                    // Assign the final calculated hitSound bitmask
                    hitObject.hitSound = baseHitSound;
                    // Assign the final calculated hitType bitmask
                    hitObject.hitType = currentHitType;
                    // console.warn(`DEBUG: Properties assigned: time=${hitObject.startTime}, type=${hitObject.hitType}, sound=${hitObject.hitSound}`);

                    // console.warn(`DEBUG: Applying defaults for object at time ${startTime}...`);
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
        // console.warn("osu string", osuString); // Keep for deep debug if needed
        console.warn("DEBUG: Encoding complete.");
        process.stdout.write(osuString); // Write final result to stdout

    } catch (err) {
        console.error(`Error during Node.js encoding process: ${err.message}`); console.error(err.stack); process.exit(1);
    }
}
encodeBeatmapFromInput();