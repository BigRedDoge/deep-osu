import os
from vocabulary import BeatmapVocabulary
from quantizers import Quantizers


class TokenToOsu:
    def __init__(self, vocab, quantizers):
        self.vocab = vocab
        self.quantizers = quantizers

    # --- .osu File Generation ---
    # ENCODE DIFFICULTY INTO MODEL
    def create_osu_file(self,
                        token_ids,
                        output_path,
                        # Metadata
                        title="Genserated Map",
                        artist="Unknown Artist",
                        creator="AI",
                        version="Normal", # Difficulty name
                        audio_filename="audio.mp3",
                        # Difficulty - Provide sensible defaults
                        hp_drain_rate=5.0,
                        circle_size=4.0,
                        overall_difficulty=5.0,
                        approach_rate=7.0,
                        slider_multiplier=1.4,
                        slider_tick_rate=1.0,
                        # Timing - A simple default (120 BPM, 4/4)
                        bpm=120.0,
                        offset=0): # Timing point offset

        hit_objects_str = []
        current_time = float(offset) # Start time based on offset
        last_x, last_y = 0, 0 # Keep track of last coords for next object

        # State variables for object construction
        object_state = None # None, "circle", "slider", "spinner"
        slider_points = [] # List of (x, y) tuples
        slider_type = None # 'L', 'B', 'P'
        slider_length = 0
        slider_start_time = 0
        spinner_start_time = 0
        new_combo = False # Flag for NEW_COMBO token

        # Object type constants from osu! format
        TYPE_CIRCLE = 1
        TYPE_SLIDER = 2
        TYPE_SPINNER = 8
        TYPE_NEW_COMBO = 4

        # --- Iterate through tokens and build hit objects ---
        for token_id in token_ids:
            token_str = self.vocab.get_token(token_id)

            if token_str is None or token_str == self.vocab.pad_token:
                continue # Ignore unknown or padding tokens
            if token_str == self.vocab.sos_token:
                 current_time = float(offset) # Reset time at SOS
                 continue
            if token_str == self.vocab.eos_token:
                break # End of sequence

            # --- Handle Token Types ---
            if token_str.startswith("TIME_SHIFT_"):
                try:
                    bin_index = int(token_str.split('_')[-1])
                    dt = self.quantizers.dequantize_time_shift(bin_index)
                    current_time += dt
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse time shift token: {token_str}")
                continue # Time shift doesn't define an object part directly

            elif token_str.startswith("COORD_X_"):
                try:
                    bin_index = int(token_str.split('_')[-1])
                    last_x = self.quantizers.dequantize_coord(bin_index, "x")
                    if object_state == "slider" and len(slider_points) == 1:
                        slider_points[0] = (last_x, last_y) # Update start point
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse X coordinate token: {token_str}")
                continue # Store coordinate for the next action token

            elif token_str.startswith("COORD_Y_"):
                try:
                    bin_index = int(token_str.split('_')[-1])
                    last_y = self.quantizers.dequantize_coord(bin_index, "y")
                    if object_state == "slider" and len(slider_points) == 1:
                        slider_points[0] = (last_x, last_y) # Update start point
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse Y coordinate token: {token_str}")
                continue # Store coordinate for the next action token

            elif token_str == "NEW_COMBO":
                new_combo = True
                # Note: NEW_COMBO applies to the *next* placed object
                continue

            elif token_str == "PLACE_CIRCLE":
                obj_type = TYPE_CIRCLE
                if new_combo:
                    obj_type |= TYPE_NEW_COMBO
                    new_combo = False # Reset flag after use

                # Format: x,y,time,type,hitSound,extras
                hit_objects_str.append(f"{int(round(last_x))},{int(round(last_y))},{int(round(current_time))},{obj_type},0") # Default hitsound 0
                object_state = None # Reset state

            elif token_str.startswith("START_SLIDER_"):
                if object_state is not None: print(f"Warning: Starting slider while in state {object_state}")
                object_state = "slider"
                slider_type = token_str.split('_')[-1] # L, B, P
                slider_points = [(last_x, last_y)] # Start point
                slider_start_time = current_time
                # New combo flag handled when slider ENDS
            elif token_str.startswith("SLIDER_DUR_"):
                if object_state != "slider":
                    print(f"Warning: Setting slider duration outside of slider state.")
                slider_length = token_str.split('_')[-1] # Duration bin index

            elif token_str == "ADD_SLIDER_ANCHOR":
                if object_state != "slider":
                    print(f"Warning: Adding slider anchor outside of slider state.")
                else:
                    slider_points.append((last_x, last_y))

            elif token_str.startswith("END_SLIDER_"):
                if object_state != "slider":
                    print(f"Warning: Ending slider outside of slider state.")
                    object_state = None
                    continue

                try:
                    repeats = int(token_str.split('_')[-1].replace('R', ''))
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse slider repeats: {token_str}")
                    repeats = 0

                # Add the final point
                slider_points.append((last_x, last_y))

                obj_type = TYPE_SLIDER
                if new_combo:
                    #obj_type |= TYPE_NEW_COMBO
                    new_combo = False

                # Format slider points string: |X1:Y1|X2:Y2...
                slider_path_str = ""
                if len(slider_points) > 1: # Need at least start + 1 anchor/end
                    slider_path_str = "|".join([f"{int(round(p[0]))}:{int(round(p[1]))}" for p in slider_points[1:]]) # Exclude start point

                # Format: x,y,time,type,hitSound,sliderType|curvePoints,repeats,pixelLength,edgeHitsounds,edgeSamplesets,extras
                # NOTE: Calculating pixelLength accurately requires slider velocity from timing points.
                # We will OMIT it here; osu! often recalculates it based on timing if missing.
                hit_objects_str.append(
                    f"{int(round(slider_points[0][0]))},{int(round(slider_points[0][1]))},{int(round(slider_start_time))},{obj_type},0," # x,y,time,type,hitSound
                    f"{slider_type}|{slider_path_str},{repeats}," # sliderType|curvePoints, repeats
                    f"{slider_length},," # Omitted pixelLength
                )

                # Reset slider state
                object_state = None
                slider_points = []
                slider_type = None
                slider_length = 0

            elif token_str == "PLACE_SPINNER":
                if object_state is not None: print(f"Warning: Placing spinner while in state {object_state}")
                object_state = "spinner"
                spinner_start_time = current_time
                 # New combo flag handled when spinner ENDS (though less common for spinners)

            elif token_str.startswith("END_SPINNER_DUR"):
                if object_state != "spinner":
                    print(f"Warning: Ending spinner outside of spinner state.")
                    object_state = None
                    continue

                try:
                    bin_index = int(token_str.split('DUR')[-1])
                    duration = self.quantizers.dequantize_spinner_duration(bin_index)
                    end_time = spinner_start_time + duration
                except (ValueError, IndexError):
                     print(f"Warning: Could not parse spinner duration: {token_str}")
                     end_time = spinner_start_time + 1000 # Default fallback duration

                obj_type = TYPE_SPINNER
                if new_combo:
                    obj_type |= TYPE_NEW_COMBO
                    new_combo = False

                # Format: x,y,time,type,hitSound,endTime,extras
                # Spinners are always centered at 256,192
                hit_objects_str.append(f"256,192,{int(round(spinner_start_time))},{obj_type},0,{int(round(end_time))}")
                object_state = None # Reset state

            else:
                print(f"Warning: Unhandled token: {token_str}")


        # --- Assemble the .osu file content ---
        beat_length = 60000.0 / bpm if bpm > 0 else 500 # ms per beat
        slider_velocity = 1.0 # Base slider velocity multiplier for timing point

        osu_content = f"""osu file format v14
[General]
AudioFilename: {os.path.basename(audio_filename)}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 0
LetterboxInBreaks: 0
WidescreenStoryboard: 1

[Editor]
DistanceSpacing: 1.0
BeatDivisor: 4
GridSize: 4
TimelineZoom: 1

[Metadata]
Title:{title}
TitleUnicode:{title}
Artist:{artist}
ArtistUnicode:{artist}
Creator:{creator}
Version:{version}
Source:
Tags:generated ai
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:{hp_drain_rate}
CircleSize:{circle_size}
OverallDifficulty:{overall_difficulty}
ApproachRate:{approach_rate}
SliderMultiplier:{slider_multiplier}
SliderTickRate:{slider_tick_rate}

[Events]
//Background and Video events
// Default black background
0,0,"",0,0

[TimingPoints]
# time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
{int(round(offset))},{beat_length},4,1,0,100,1,0

[HitObjects]
"""
        osu_content += "\n".join(hit_objects_str)
        osu_content += "\n" # Ensure trailing newline

        # --- Write to file ---
        try:
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                f.write(osu_content)
            print(f"Successfully created .osu file at: {output_path}")
        except IOError as e:
            print(f"Error writing .osu file: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during file writing: {e}")