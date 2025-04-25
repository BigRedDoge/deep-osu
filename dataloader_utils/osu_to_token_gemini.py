# dataloader_utils/osu_to_token.py
import json
import subprocess
import os
import sys
from pathlib import Path

# Assuming quantizers.py and vocabulary.py are in the same directory
# If not, adjust the import path accordingly
try:
    from quantizers import Quantizers # Example class name, adjust if needed
    from vocabulary import BeatmapVocabulary
except ImportError:
    # Allow running script directly for testing
    sys.path.append(str(Path(__file__).parent))
    from quantizers import QuantizationManager
    from vocabulary import BeatmapVocabulary


class OsuToToken:
    def __init__(self, osu_file_path, quantizers, vocab, max_seq_len=20480, node_parser_script="parser/parse_osu.js"):
        """
        Initializes the tokenizer.

        Args:
            osu_file_path (str): Path to the .osu file.
            quantizers (QuantizationManager): Instance for value quantization.
            vocab (BeatmapVocabulary): Instance for token mapping.
            max_seq_len (int): Maximum sequence length for padding/truncation.
            node_parser_script (str): Relative path to the Node.js parser script.
        """
        self.osu_file_path = osu_file_path
        self.quantizers = quantizers
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        # Determine the absolute path to the node script relative to this file's location
        # Assumes node_parser_script is relative to the project root or a known base
        script_dir = Path(__file__).parent.parent # Assumes this script is in dataloader_utils
        self.node_parser_path = script_dir / node_parser_script
        if not self.node_parser_path.is_file():
             # Fallback if the above assumption is wrong, try relative to cwd
             self.node_parser_path = Path(node_parser_script).resolve()
             if not self.node_parser_path.is_file():
                 raise FileNotFoundError(f"Node parser script not found at expected locations: {script_dir / node_parser_script} or {Path(node_parser_script).resolve()}")

        self.parsed_data = self._run_node_parser()

        # Initialize the sequence with Start Of Sequence token
        self.token_id_sequence = [vocab.sos_id]
        if self.parsed_data and 'hitObjects' in self.parsed_data:
            self.tokenize_osu()
        else:
             print(f"Warning: No hit objects found or error parsing file: {self.osu_file_path}")
             # Ensure sequence ends correctly even if empty/error
             self._finalize_and_pad()


    def _run_node_parser(self):
        """Executes the Node.js parser script and returns the parsed JSON data."""
        try:
            # Ensure Node.js is installed and executable
            # Use absolute paths for robustness
            print(self.osu_file_path[0])
            osu_abs_path = str(Path(self.osu_file_path[0]).resolve())
            print(osu_abs_path)
            node_script_abs_path = str(self.node_parser_path.resolve())

            # Make sure node script has execute permissions if needed (less common)
            # os.chmod(node_script_abs_path, 0o755)

            command = ['node', node_script_abs_path, osu_abs_path]
            # print(f"Running node command: {' '.join(command)}") # Debugging

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding='utf-8', # Be explicit about encoding
                # check=True, # Temporarily disable check=True to inspect output even on failure
                cwd=self.node_parser_path.parent
            )

            # --- Step 3: Inspect ALL output regardless of success ---
            print(f"DEBUG: Node process finished. Return code: {result.returncode}")
            print(f"DEBUG: Raw stdout (length {len(result.stdout)}):\n'''\n{result.stdout}\n'''")
            print(f"DEBUG: Raw stderr (length {len(result.stderr)}):\n'''\n{result.stderr}\n'''")

            # Handle potential warnings or extra output from node before JSON
            json_output = result.stdout.strip()
            #print(f"Node output: {json_output}") # Debugging output
            # Find the start of the JSON object (robustness against warnings)
            json_start_index = json_output.find('{')
            #if not result.stdout.strip().startswith('{'):
            #    print("Error: Node stdout does not start with '{'. Content might be corrupted or not JSON.")
            #    return None
            if json_start_index == -1:
                 print(f"Error: No JSON object found in Node output for {self.osu_file_path}.")
                 #print(f"Node stdout: {result.stdout}")
                 #print(f"Node stderr: {result.stderr}")
                 return None

            json_output = json_output[json_start_index:]
            print(f"DEBUG: JSON output (length {len(json_output)}):\n'")
            #print(f"Parsed JSON output: {json_output}") # Debugging output
            return json.loads(json_output)
            #return result.stdout.strip()

        except FileNotFoundError:
            print("Error: 'node' command not found. Please ensure Node.js is installed and in your PATH.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error running Node.js parser for {self.osu_file_path}:")
            print(f"  Return code: {e.returncode}")
            #print(f"  Stdout: {e.stdout}")
            #print(f"  Stderr: {e.stderr}")
            # Try to parse JSON even from stderr if stdout is empty
            try:
                error_json = json.loads(e.stdout.strip()[e.stdout.strip().find('{'):])
                return error_json # Return error structure if available
            except json.JSONDecodeError:
                 return None # Failed to parse JSON error output
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON output from Node.js for {self.osu_file_path}: {e}")
            #print(f"Raw Node output: {result.stdout}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while running the Node parser: {e}")
            return None


    def _append_token(self, token_str):
        """Appends a token ID to the sequence, checking length limits."""
        # Check if we have space (leave 1 for EOS)
        if len(self.token_id_sequence) >= self.max_seq_len - 1:
            # Don't print warning here, handle truncation at the end
            return False # Signal that sequence is full

        token_id = self.vocab.get_id(token_str)
        if token_id is None:
            print(f"Warning: Token not found in vocabulary for: '{token_str}'. Skipping.")
            # Decide how to handle unknown tokens: skip, use an <UNK> token, or raise error
            return True # Allow sequence building to continue, but token is skipped
        else:
            self.token_id_sequence.append(token_id)
            return True


    def _finalize_and_pad(self):
        """Handles EOS token appending, truncation, and padding."""
        if not self.token_id_sequence or self.token_id_sequence[-1] == self.vocab.sos_id:
            # Handle empty or only SOS sequence case - add EOS if space
             if self.max_seq_len > 1:
                self.token_id_sequence = [self.vocab.sos_id, self.vocab.eos_id]
             else: # Not enough space even for SOS, EOS
                self.token_id_sequence = [self.vocab.sos_id] # Or maybe just PAD? Depends on model reqs.


        # If sequence reached max length *before* adding EOS
        elif len(self.token_id_sequence) >= self.max_seq_len:
            # Truncate and force the last token to be EOS
            self.token_id_sequence = self.token_id_sequence[:self.max_seq_len]
            self.token_id_sequence[-1] = self.vocab.eos_id
            print(f"Warning: Sequence truncated to {self.max_seq_len} tokens for {self.osu_file_path}.")

        # If sequence is shorter than max length and doesn't end with EOS
        elif self.token_id_sequence[-1] != self.vocab.eos_id:
            self.token_id_sequence.append(self.vocab.eos_id)

        # Pad the sequence if it's still shorter than max_seq_len
        padding_needed = self.max_seq_len - len(self.token_id_sequence)
        if padding_needed > 0:
            self.token_id_sequence.extend([self.vocab.pad_id] * padding_needed)

        # Final ensure length is exactly max_seq_len (should be redundant now)
        self.token_id_sequence = self.token_id_sequence[:self.max_seq_len]


    def tokenize_osu(self):
        """Tokenizes the parsed osu data according to plan.md."""
        last_event_time_ms = 0
        sequence_full = False

        # Sort hit objects by start time just in case node parser doesn't guarantee it
        hit_objects = sorted(self.parsed_data.get('hitObjects', []), key=lambda ho: ho['startTime'])

        for hit_object in hit_objects:
            if sequence_full:
                break

            # --- 1. Time Shift ---
            current_event_time_ms = hit_object['startTime']
            time_difference_ms = current_event_time_ms - last_event_time_ms
            time_difference_ms = max(0, time_difference_ms) # Ensure non-negative

            # Assuming quantize_time_shift returns a single bin index
            time_bin = self.quantizers.quantize_time_shift(time_difference_ms)
            if not self._append_token(f"TIME_SHIFT_{time_bin}"):
                sequence_full = True; continue

            # --- 2. New Combo ---
            # Using the isNewCombo flag provided by the parser
            if hit_object.get('isNewCombo', False):
                 if not self._append_token("NEW_COMBO"):
                     sequence_full = True; continue

            # --- 3. Object Specific Tokens ---
            object_type = hit_object['objectType']

            # --- 3a. Circle ---
            if object_type == 'Circle':
                if not self._append_token("PLACE_CIRCLE"):
                    sequence_full = True; continue

                # Normalize and quantize absolute coordinates
                norm_x = hit_object['startX'] / 512.0
                norm_y = hit_object['startY'] / 384.0
                x_bin = self.quantizers.quantize_coord(norm_x, 'x')
                y_bin = self.quantizers.quantize_coord(norm_y, 'y')

                if not self._append_token(f"COORD_X_{x_bin}"): sequence_full = True; continue
                if not self._append_token(f"COORD_Y_{y_bin}"): sequence_full = True; continue

            # --- 3b. Slider ---
            elif object_type == 'Slider':
                slider_path = hit_object.get('path')
                if not slider_path:
                     print(f"Warning: Slider object at {current_event_time_ms}ms missing path data. Skipping.")
                     last_event_time_ms = current_event_time_ms # Still update time
                     continue

                curve_type_char = slider_path.get('curveTypeChar', 'B') # Default to Bezier if missing
                # Handle Catmull ('C') by mapping to Bezier ('B') as per plan
                if curve_type_char == 'C':
                    curve_type_char = 'B'

                start_token = f"START_SLIDER_{curve_type_char}"
                if not self._append_token(start_token): sequence_full = True; continue

                # Tokenize start position (absolute)
                norm_x = hit_object['startX'] / 512.0
                norm_y = hit_object['startY'] / 384.0
                start_x_bin = self.quantizers.quantize_coord(norm_x, 'x')
                start_y_bin = self.quantizers.quantize_coord(norm_y, 'y')
                if not self._append_token(f"COORD_X_{start_x_bin}"): sequence_full = True; continue
                if not self._append_token(f"COORD_Y_{start_y_bin}"): sequence_full = True; continue

                # Tokenize Anchors (Relative Coordinates) - plan.md Step 1.3
                control_points = slider_path.get('controlPoints', [])
                last_point_abs = {'x': hit_object['startX'], 'y': hit_object['startY']}

                # Iterate from the second point onwards (index 1)
                for i in range(1, len(control_points)):
                    anchor_point_abs = control_points[i]

                    # Calculate relative delta (dx, dy)
                    # Note: osu-parsers gives absolute coords for anchors
                    dx = anchor_point_abs['x'] - last_point_abs['x']
                    dy = anchor_point_abs['y'] - last_point_abs['y']

                    # Quantize relative coordinates (needs quantize_relative_coord in QuantizationManager)
                    # Assuming it takes raw pixel deltas
                    dx_bin = self.quantizers.quantize_relative_coord(dx, 'x')
                    dy_bin = self.quantizers.quantize_relative_coord(dy, 'y')

                    if not self._append_token("ADD_SLIDER_ANCHOR"): sequence_full = True; break # Break inner loop
                    if not self._append_token(f"COORD_X_{dx_bin}"): sequence_full = True; break
                    if not self._append_token(f"COORD_Y_{dy_bin}"): sequence_full = True; break

                    # Update last absolute position for next delta calculation
                    last_point_abs = anchor_point_abs
                if sequence_full: continue # Continue outer loop if inner loop broke

                # Tokenize slider end (repeats)
                repeats = hit_object.get('repeats', 0)
                # Clamp repeats to max value defined in vocabulary
                repeats_clamped = min(repeats, self.vocab.max_slider_repeats)
                # Assuming quantize_slider_repeats just returns the clamped value directly
                repeat_bin = self.quantizers.quantize_slider_repeats(repeats_clamped)

                if not self._append_token(f"END_SLIDER_{repeat_bin}R"):
                     sequence_full = True; continue

                # --- REMOVED: SLIDER_DUR tokenization as per plan.md ---
                # slider_dur_bin = self.quantizers.quantize_slider_duration(hit_object.length) # 'length' not provided by node parser directly
                # if not self._append_token(f"SLIDER_DUR_{slider_dur_bin}"): break

            # --- 3c. Spinner ---
            elif object_type == 'Spinner':
                if not self._append_token("PLACE_SPINNER"): sequence_full = True; continue

                end_time = hit_object.get('endTime', current_event_time_ms) # Use start time if end time missing
                duration_ms = end_time - current_event_time_ms
                duration_ms = max(0, duration_ms) # Ensure non-negative

                duration_bin = self.quantizers.quantize_spinner_duration(duration_ms)
                if not self._append_token(f"END_SPINNER_DUR{duration_bin}"):
                     sequence_full = True; continue

            # --- 3d. Unknown ---
            else:
                print(f"Warning: Unknown object type '{object_type}' encountered at time {current_event_time_ms}. Skipping.")

            # Update time for the next iteration
            last_event_time_ms = current_event_time_ms

        # --- 4. Finalize Sequence (EOS, Padding/Truncation) ---
        self._finalize_and_pad()

        return self.token_id_sequence

    def get_token_sequence(self):
        """Returns the final token ID sequence."""
        # Ensure tokenization has run
        if not hasattr(self, 'token_id_sequence'):
             self.tokenize_osu() # Run if not already done (e.g., if init failed)
        return self.token_id_sequence


# Example Usage (for testing)
if __name__ == "__main__":
    
    test_osu_path = "beatmapparser/harmony/harmony-hard.osu",

    # --- Test the tokenizer ---
    print(f"\nTesting tokenizer with: {test_osu_path}")
    vocab = BeatmapVocabulary()
    quantizer = Quantizers(vocab)

    # Provide path to the *actual* node script relative to this test script's execution context
    # Adjust this relative path based on your project structure
    # Example: If running osu_to_token.py directly, and parser/ is sibling to dataloader_utils/
    node_script_relative_path = "../parser/parse_osu_gemini.js"


    try:
        # Pass the relative path from the script's perspective
        tokenizer = OsuToToken(test_osu_path, quantizer, vocab, max_seq_len=4096, node_parser_script=node_script_relative_path)
        final_sequence = tokenizer.get_token_sequence()
        print("final", final_sequence)
        print(f"\nGenerated Token ID Sequence (len={len(final_sequence)}):")
        print(final_sequence)

        print("\nDecoded Tokens:")
        decoded_tokens = [vocab.id_to_token.get(tid, f"ID_{tid}_?") for tid in final_sequence]
        print(" \n".join(decoded_tokens))

    except FileNotFoundError as e:
         print(f"\nError during test: {e}")
         print("Ensure the test .osu file exists and the node parser script path is correct relative to where you run this Python script.")
    except Exception as e:
         print(f"\nAn error occurred during testing: {e}")