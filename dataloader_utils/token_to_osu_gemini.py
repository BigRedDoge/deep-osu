# dataloader_utils/token_to_osu.py
import json
import subprocess
from pathlib import Path
import sys
import math # Needed for coordinate dequantization

# Assuming quantizers.py and vocabulary.py are in the same directory or accessible
try:
    from .quantizers import QuantizationManager # Adjust class name if needed
    from .vocabulary import BeatmapVocabulary
except ImportError:
    # Allow running script directly for testing
    sys.path.append(str(Path(__file__).parent))
    from quantizers import QuantizationManager
    from vocabulary import BeatmapVocabulary

class TokenToOsu:
    def __init__(self, quantizers, vocab, node_encoder_script="encoder/encode_osu.js"):
        """
        Initializes the detokenizer.

        Args:
            quantizers (QuantizationManager): Instance for value dequantization.
            vocab (BeatmapVocabulary): Instance for token mapping.
            node_encoder_script (str): Relative path to the Node.js encoder script.
        """
        self.quantizers = quantizers
        self.vocab = vocab

        # Determine the absolute path to the node script
        script_dir = Path(__file__).parent.parent # Assumes this script is in dataloader_utils
        self.node_encoder_path = script_dir / node_encoder_script
        if not self.node_encoder_path.is_file():
             self.node_encoder_path = Path(node_encoder_script).resolve()
             if not self.node_encoder_path.is_file():
                 raise FileNotFoundError(f"Node encoder script not found at expected locations: {script_dir / node_encoder_script} or {Path(node_encoder_script).resolve()}")

        # Internal state during detokenization
        self._reset_state()

    def _reset_state(self):
        """Resets the internal state for processing a new sequence."""
        self.hit_objects = []
        self.current_time_ms = 0
        self.pending_object_type = None
        self.pending_x_bin = None
        self.pending_y_bin = None
        self.new_combo_pending = False
        self.current_slider_data = None # Stores {'type': 'L'/'B'/'C'/'P', 'start_time': ms, 'points': [{'x':px, 'y':px}, ...], 'repeats': n}
        self.spinner_start_time = 0

    def detokenize(self, token_id_sequence):
        """
        Converts a sequence of token IDs into a list of hit object dictionaries.

        Args:
            token_id_sequence (list[int]): The sequence of token IDs.

        Returns:
            list[dict]: A list of dictionaries, each representing a hit object.
        """
        self._reset_state()

        # Remove padding and SOS, find EOS
        try:
            eos_index = token_id_sequence.index(self.vocab.eos_id)
            processed_sequence = token_id_sequence[1:eos_index] # Skip SOS, stop before EOS
        except ValueError:
            # No EOS found, process up to the first PAD or end
            try:
                pad_index = token_id_sequence.index(self.vocab.pad_id)
                processed_sequence = token_id_sequence[1:pad_index] # Skip SOS
            except ValueError:
                processed_sequence = token_id_sequence[1:] # Skip SOS, process all

        if not processed_sequence:
             print("Warning: Empty sequence after removing SOS/EOS/PAD.")
             return []


        for token_id in processed_sequence:
            token_str = self.vocab.get_token(token_id)

            if token_str is None:
                print(f"Warning: Unknown token ID {token_id} encountered. Skipping.")
                continue

            # --- Handle Time Shift ---
            if token_str.startswith("TIME_SHIFT_"):
                try:
                    time_bin = int(token_str.split('_')[-1])
                    dt_ms = self.quantizers.dequantize_time_shift(time_bin)
                    self.current_time_ms += dt_ms
                except (ValueError, IndexError, AttributeError) as e:
                    print(f"Error processing time token '{token_str}': {e}")
                except TypeError:
                     print(f"Error: dequantize_time_shift not implemented or failed for bin {time_bin}")


            # --- Handle New Combo ---
            elif token_str == "NEW_COMBO":
                self.new_combo_pending = True

            # --- Handle Object Start/Type Tokens ---
            elif token_str == "PLACE_CIRCLE":
                self._finalize_previous_object() # Finalize any pending slider/spinner
                self.pending_object_type = 'circle'
            elif token_str.startswith("START_SLIDER_"):
                self._finalize_previous_object()
                self.pending_object_type = 'slider_start'
                slider_type = token_str[-1] # 'L', 'B', 'C', or 'P'
                self.current_slider_data = {
                    'type': slider_type,
                    'start_time': self.current_time_ms,
                    'points': [],
                    'repeats': 0, # Default, will be set by END_SLIDER
                    'is_new_combo': self.new_combo_pending # Capture combo state at start
                }
                self.new_combo_pending = False # Consume flag for this object
            elif token_str == "ADD_SLIDER_ANCHOR":
                if self.current_slider_data and self.current_slider_data['points']:
                     self.pending_object_type = 'slider_anchor'
                else:
                    print(f"Warning: Encountered ADD_SLIDER_ANCHOR without a valid preceding slider start/point at time {self.current_time_ms}. Skipping anchor.")
                    self.pending_object_type = None # Prevent coordinate processing
            elif token_str == "PLACE_SPINNER":
                self._finalize_previous_object()
                self.pending_object_type = 'spinner'
                self.spinner_start_time = self.current_time_ms

            # --- Handle Coordinate Tokens ---
            elif token_str.startswith("COORD_X_"):
                try:
                    self.pending_x_bin = int(token_str.split('_')[-1])
                    # If Y was already pending, process coordinates now
                    if self.pending_y_bin is not None:
                        self._process_coordinates()
                except (ValueError, IndexError) as e:
                     print(f"Error processing X coordinate token '{token_str}': {e}")
                     self.pending_x_bin = None # Reset on error

            elif token_str.startswith("COORD_Y_"):
                try:
                    self.pending_y_bin = int(token_str.split('_')[-1])
                    # If X was already pending, process coordinates now
                    if self.pending_x_bin is not None:
                         self._process_coordinates()
                except (ValueError, IndexError) as e:
                     print(f"Error processing Y coordinate token '{token_str}': {e}")
                     self.pending_y_bin = None # Reset on error


            # --- Handle Object End Tokens ---
            elif token_str.startswith("END_SLIDER_"):
                if self.current_slider_data:
                    try:
                        repeat_bin_str = token_str.split('_')[-1][:-1] # Get number part
                        repeat_bin = int(repeat_bin_str)
                        self.current_slider_data['repeats'] = self.quantizers.dequantize_slider_repeats(repeat_bin)
                        self._finalize_slider() # Add slider to hit_objects list
                    except (ValueError, IndexError, AttributeError) as e:
                        print(f"Error processing slider end token '{token_str}': {e}. Discarding slider.")
                        self.current_slider_data = None # Discard incomplete slider
                    except TypeError:
                         print(f"Error: dequantize_slider_repeats not implemented or failed for bin {repeat_bin}. Discarding slider.")
                         self.current_slider_data = None
                else:
                    print(f"Warning: Encountered END_SLIDER token without active slider data at time {self.current_time_ms}. Ignoring.")
                self.pending_object_type = None # Ensure state is reset

            elif token_str.startswith("END_SPINNER_DUR"):
                 if self.pending_object_type == 'spinner':
                    try:
                        duration_bin_str = token_str[len("END_SPINNER_DUR"):]
                        duration_bin = int(duration_bin_str)
                        duration_ms = self.quantizers.dequantize_spinner_duration(duration_bin)
                        end_time = self.spinner_start_time + duration_ms

                        spinner_obj = {
                            'object_type': 'Spinner',
                            'time': self.spinner_start_time,
                            'end_time': end_time,
                            'x': 256, # Spinners are always centered
                            'y': 192,
                            'is_new_combo': self.new_combo_pending
                        }
                        self.hit_objects.append(spinner_obj)
                        self.new_combo_pending = False # Consume flag
                        self.pending_object_type = None
                    except (ValueError, IndexError, AttributeError) as e:
                        print(f"Error processing spinner end token '{token_str}': {e}. Discarding spinner.")
                        self.pending_object_type = None
                    except TypeError:
                         print(f"Error: dequantize_spinner_duration not implemented or failed for bin {duration_bin}. Discarding spinner.")
                         self.pending_object_type = None
                 else:
                     print(f"Warning: Encountered END_SPINNER token without active spinner data at time {self.current_time_ms}. Ignoring.")
                     self.pending_object_type = None

        # Final check for any object pending after loop finishes
        self._finalize_previous_object()

        return self.hit_objects

    def _process_coordinates(self):
        """Processes pending coordinate bins based on the pending object type."""
        if self.pending_x_bin is None or self.pending_y_bin is None:
            # Should not happen if called correctly, but safety check
            return

        try:
            if self.pending_object_type == 'circle':
                abs_x = self.quantizers.dequantize_coord(self.pending_x_bin, 'x')
                abs_y = self.quantizers.dequantize_coord(self.pending_y_bin, 'y')

                circle_obj = {
                    'object_type': 'Circle',
                    'time': self.current_time_ms,
                    'x': abs_x,
                    'y': abs_y,
                    'is_new_combo': self.new_combo_pending
                }
                self.hit_objects.append(circle_obj)
                self.new_combo_pending = False # Consume flag

            elif self.pending_object_type == 'slider_start':
                 if self.current_slider_data:
                    abs_x = self.quantizers.dequantize_coord(self.pending_x_bin, 'x')
                    abs_y = self.quantizers.dequantize_coord(self.pending_y_bin, 'y')
                    self.current_slider_data['points'].append({'x': abs_x, 'y': abs_y})
                 else:
                     print("Warning: Tried to process slider start coordinates without slider data.")


            elif self.pending_object_type == 'slider_anchor':
                if self.current_slider_data and self.current_slider_data['points']:
                    # Dequantize relative delta
                    dx = self.quantizers.dequantize_relative_coord(self.pending_x_bin, 'x')
                    dy = self.quantizers.dequantize_relative_coord(self.pending_y_bin, 'y')

                    # Get last absolute point
                    last_point = self.current_slider_data['points'][-1]
                    new_x = last_point['x'] + dx
                    new_y = last_point['y'] + dy
                    self.current_slider_data['points'].append({'x': new_x, 'y': new_y})
                else:
                    print("Warning: Tried to process slider anchor coordinates without valid slider data/points.")

            # Reset pending state after processing
            self.pending_object_type = None
            self.pending_x_bin = None
            self.pending_y_bin = None

        except AttributeError as e:
             print(f"Error during coordinate dequantization: {e}. Check QuantizationManager methods.")
             # Reset state to prevent cascading errors
             self.pending_object_type = None
             self.pending_x_bin = None
             self.pending_y_bin = None
        except TypeError as e:
             print(f"Error calling dequantization method: {e}. Check QuantizationManager methods.")
             self.pending_object_type = None
             self.pending_x_bin = None
             self.pending_y_bin = None


    def _finalize_slider(self):
        """Adds the completed slider data to the hit objects list."""
        if self.current_slider_data:
            if len(self.current_slider_data['points']) >= 2: # Need at least start and end
                slider_obj = {
                    'object_type': 'Slider',
                    'time': self.current_slider_data['start_time'],
                    'x': self.current_slider_data['points'][0]['x'], # Start position
                    'y': self.current_slider_data['points'][0]['y'],
                    'is_new_combo': self.current_slider_data['is_new_combo'],
                    'slider_type_char': self.current_slider_data['type'],
                    'repeats': self.current_slider_data['repeats'],
                    'points': self.current_slider_data['points'] # Pass all points
                }
                self.hit_objects.append(slider_obj)
            else:
                print(f"Warning: Discarding slider starting at {self.current_slider_data['start_time']} due to insufficient points ({len(self.current_slider_data['points'])}).")
        self.current_slider_data = None # Reset slider state

    def _finalize_previous_object(self):
        """Finalizes any pending slider before starting a new object."""
        if self.current_slider_data:
            print(f"Warning: Finalizing incomplete slider starting at {self.current_slider_data['start_time']} due to new object start.")
            self._finalize_slider()
        # Could add similar logic for spinners if they weren't ended explicitly


    def get_osu_string(self, token_id_sequence, metadata_overrides=None, difficulty_overrides=None, timing_point_overrides=None):
        """
        Detokenizes the sequence and calls the Node.js encoder to get the .osu string.

        Args:
            token_id_sequence (list[int]): The sequence of token IDs.
            metadata_overrides (dict, optional): Dictionary to override default metadata.
            difficulty_overrides (dict, optional): Dictionary to override default difficulty.
            timing_point_overrides (dict, optional): Dictionary to override default timing point.

        Returns:
            str: The generated .osu file content as a string, or None on error.
        """
        hit_object_list = self.detokenize(token_id_sequence)

        if not hit_object_list and not token_id_sequence: # Handle empty input case
             print("Input token sequence is empty, cannot generate .osu string.")
             # Optionally return a minimal valid .osu string
             # return self._get_minimal_osu_string(metadata_overrides, difficulty_overrides, timing_point_overrides)
             return None


        # Prepare JSON payload for the Node.js encoder
        payload = {
            'metadata': metadata_overrides or {}, # Use defaults if None
            'difficulty': difficulty_overrides or {},
            'timing_point': timing_point_overrides or {}, # Use defaults if None
            'hit_objects': hit_object_list
        }
        payload_json = json.dumps(payload)

        try:
            # Ensure Node.js is installed and executable
            node_script_abs_path = str(self.node_encoder_path.resolve())

            command = ['node', node_script_abs_path]
            # print(f"Running node command: {' '.join(command)}") # Debugging
            # print(f"Input JSON:\n{payload_json[:500]}...") # Debugging input

            result = subprocess.run(
                command,
                input=payload_json, # Pass JSON via stdin
                capture_output=True,
                text=True,
                check=True, # Raise exception on non-zero exit code
                cwd=self.node_encoder_path.parent # Run node from the encoder's directory
            )

            osu_string = result.stdout.strip()
            if not osu_string.startswith("osu file format"):
                 print("Error: Node encoder did not produce a valid .osu string.")
                 print(f"Node stdout: {result.stdout}")
                 print(f"Node stderr: {result.stderr}")
                 return None

            return osu_string

        except FileNotFoundError:
            print("Error: 'node' command not found. Please ensure Node.js is installed and in your PATH.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error running Node.js encoder:")
            print(f"  Return code: {e.returncode}")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while running the Node encoder: {e}")
            return None

    # Optional: Helper for empty output
    # def _get_minimal_osu_string(self, meta=None, diff=None, timing=None):
    #     m = meta or {}
    #     d = diff or {}
    #     t = timing or {}
    #     # ... construct minimal valid osu string ...
    #     return "osu file format v14\n[General]\nMode: 0\n[Metadata]\nTitle:Generated\nArtist:AI\nCreator:AI\nVersion:Empty\n[Difficulty]\n..."


# Example Usage (Requires updated DummyQuantizers and Node encoder script)
if __name__ == "__main__":
    # --- Dummy Classes (Update Quantizers) ---
    class DummyVocab: # (Use the one from osu_to_token example or load a real one)
        def __init__(self):
            self.pad_token = "<PAD>"
            self.sos_token = "<SOS>"
            self.eos_token = "<EOS>"
            self.token_to_id = {
                "<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "TIME_SHIFT_10": 3, "NEW_COMBO": 4,
                "PLACE_CIRCLE": 5, "COORD_X_15": 6, "COORD_Y_20": 7,
                "START_SLIDER_B": 8, "ADD_SLIDER_ANCHOR": 9, "COORD_X_5": 10, "COORD_Y_2": 11,
                "END_SLIDER_1R": 12, "PLACE_SPINNER": 13, "END_SPINNER_DUR5": 14,
                 "START_SLIDER_L": 15, "START_SLIDER_P": 16,
                 "TIME_SHIFT_5": 17, "TIME_SHIFT_0": 18,
                 "COORD_X_0": 19, "COORD_Y_0": 20, "COORD_X_31": 21, "COORD_Y_23": 22,
                 "END_SLIDER_0R": 23, "END_SLIDER_4R": 24,
                 "END_SPINNER_DUR0": 25, "END_SPINNER_DUR15": 26,
            }
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            self.vocab_size = len(self.token_to_id)
            self.pad_id = 0
            self.sos_id = 1
            self.eos_id = 2
            self.max_slider_repeats = 4
            self.coord_x_bins = 32
            self.coord_y_bins = 24
            self.time_shift_bins = 64
            self.spinner_duration_bins = 16
            self.slider_max_relative_delta = 128.0


        def get_id(self, token_str): return self.token_to_id.get(token_str, None)
        def get_token(self, token_id): return self.id_to_token.get(token_id, None)
        def __len__(self): return self.vocab_size

    class DummyQuantizers: # ADD DEQUANTIZE METHODS
        def __init__(self, vocab=None):
             # Store vocab reference if needed for bin counts etc.
             self.vocab = vocab if vocab else DummyVocab() # Use dummy if none passed

        def _get_bin_midpoint(self, bin_index, total_bins, min_val, max_val):
            """Helper to get the midpoint value for a linear quantization bin."""
            bin_width = (max_val - min_val) / total_bins
            bin_start = min_val + bin_index * bin_width
            return bin_start + bin_width / 2

        def dequantize_time_shift(self, bin_index):
            # Simple inverse of example quantize - needs to match actual logic
            # This is a very rough approximation
            if bin_index == 0: return 0
            # Assume roughly exponential bins, find midpoint
            # This needs to be the inverse of the *actual* quantize_time_shift
            # Example: if quantize was roughly log based
            min_time = 1 # ms
            max_time = 5000 # ms - adjust based on actual range
            log_min = math.log(min_time) if min_time > 0 else 0
            log_max = math.log(max_time)
            log_val = log_min + (bin_index + 0.5) * (log_max - log_min) / self.vocab.time_shift_bins
            return math.exp(log_val)
            # return float(bin_index * 100 + 50) # Very basic linear midpoint

        def dequantize_coord(self, bin_index, axis):
            bins = self.vocab.coord_x_bins if axis == 'x' else self.vocab.coord_y_bins
            # Map bin index back to normalized 0-1 range (midpoint)
            return self._get_bin_midpoint(bin_index, bins, 0.0, 1.0)
            # return (bin_index + 0.5) / bins

        def dequantize_relative_coord(self, bin_index, axis):
            bins = self.vocab.coord_x_bins if axis == 'x' else self.vocab.coord_y_bins
            max_delta = self.vocab.slider_max_relative_delta
            # Map bin index back to 0-1 range
            norm_01 = self._get_bin_midpoint(bin_index, bins, 0.0, 1.0)
            # Map 0-1 range back to -1 to 1 range
            norm_delta = (norm_01 * 2.0) - 1.0
            # Scale back to pixel delta
            return norm_delta * max_delta


        def dequantize_slider_repeats(self, bin_index):
            return bin_index # Repeats are stored directly as bin index

        def dequantize_spinner_duration(self, bin_index):
            # Simple inverse of example quantize - needs to match actual logic
            # return float(bin_index * 500 + 250) # Linear midpoint example
            return self._get_bin_midpoint(bin_index, self.vocab.spinner_duration_bins, 0, 8000) # Example range 0-8 sec


    # --- Test ---
    print("\nTesting TokenToOsu...")
    vocab_instance = DummyVocab()
    quantizer_instance = DummyQuantizers(vocab_instance)

    # Example token sequence (Matches the dummy test_beatmap.osu from osu_to_token)
    # SOS, TIME_SHIFT(500), CIRCLE, COORD_X(15), COORD_Y(9),
    # TIME_SHIFT(500), START_SLIDER_B, COORD_X(6), COORD_Y(4), ADD_ANCHOR, COORD_X(relative?), COORD_Y(relative?), ADD_ANCHOR, COORD_X(rel?), COORD_Y(rel?), END_SLIDER_1R,
    # TIME_SHIFT(1000), SPINNER, END_SPINNER_DUR(3),
    # TIME_SHIFT(2000), NEW_COMBO, CIRCLE, COORD_X(24), COORD_Y(15), EOS, PAD, PAD...

    # Approximate token IDs based on DummyVocab
    # Note: Relative coord bins depend heavily on the actual deltas in the test file and quantizer logic
    # These are placeholders and likely incorrect for slider anchors
    test_token_ids = [
        1, # SOS
        vocab_instance.get_id("TIME_SHIFT_5"), # 500ms -> bin 5 approx
        vocab_instance.get_id("PLACE_CIRCLE"),
        vocab_instance.get_id("COORD_X_15"), # 256 -> bin 15
        vocab_instance.get_id("COORD_Y_9"),  # 192 -> bin 9
        vocab_instance.get_id("TIME_SHIFT_5"), # 500ms -> bin 5
        vocab_instance.get_id("START_SLIDER_B"),
        vocab_instance.get_id("COORD_X_6"),  # 100 -> bin 6
        vocab_instance.get_id("COORD_Y_4"),  # 100 -> bin 4
        # --- Slider Anchors (Need actual relative quantization) ---
        # Point 1: 200,200 -> Delta from 100,100 is 100,100
        vocab_instance.get_id("ADD_SLIDER_ANCHOR"),
        vocab_instance.get_id("COORD_X_27"), # Quantize(100, 'x') -> High bin? Example: 27
        vocab_instance.get_id("COORD_Y_20"), # Quantize(100, 'y') -> High bin? Example: 20
        # Point 2: 300,100 -> Delta from 200,200 is 100,-100
        vocab_instance.get_id("ADD_SLIDER_ANCHOR"),
        vocab_instance.get_id("COORD_X_27"), # Quantize(100, 'x') -> Example: 27
        vocab_instance.get_id("COORD_Y_3"),  # Quantize(-100, 'y') -> Low bin? Example: 3
        # --- End Slider Anchors ---
        vocab_instance.get_id("END_SLIDER_1R"), # 1 repeat
        vocab_instance.get_id("TIME_SHIFT_10"),# 1000ms -> bin 10
        vocab_instance.get_id("PLACE_SPINNER"),
        vocab_instance.get_id("END_SPINNER_DUR3"),# Duration 1500ms -> bin 3 approx
        vocab_instance.get_id("TIME_SHIFT_5"), # 500ms -> bin 5 (relative to spinner *start*)
        vocab_instance.get_id("NEW_COMBO"),
        vocab_instance.get_id("PLACE_CIRCLE"),
        vocab_instance.get_id("COORD_X_24"), # 400 -> bin 24
        vocab_instance.get_id("COORD_Y_15"), # 300 -> bin 15
        2, # EOS
        0, 0, 0 # PAD
    ]
    test_token_ids = [tid for tid in test_token_ids if tid is not None] # Remove None if get_id failed

    # Provide path to the *actual* node script relative to this test script's execution context
    # Adjust this relative path based on your project structure
    # Example: If running token_to_osu.py directly, and encoder/ is sibling to dataloader_utils/
    node_encoder_relative_path = "../encoder/encode_osu.js" # Assumes encoder dir exists

    # Check if encoder script exists before proceeding
    encoder_script_full_path = Path(__file__).parent.parent / node_encoder_relative_path
    if not encoder_script_full_path.is_file():
         encoder_script_full_path = Path(node_encoder_relative_path).resolve()

    if encoder_script_full_path.is_file():
        try:
            detokenizer = TokenToOsu(quantizer_instance, vocab_instance, node_encoder_script=node_encoder_relative_path)
            print("\nDetokenizing sequence...")
            reconstructed_objects = detokenizer.detokenize(test_token_ids) # Run detokenize first
            print(f"Reconstructed {len(reconstructed_objects)} hit objects.")
            # for i, obj in enumerate(reconstructed_objects):
            #     print(f"  {i}: {obj}")

            print("\nGenerating .osu string via Node.js encoder...")
            osu_output_string = detokenizer.get_osu_string(test_token_ids) # Pass sequence again to get_osu_string

            if osu_output_string:
                print("\nGenerated .osu String (Partial):")
                print(osu_output_string[:1000] + "\n...") # Print first 1000 chars

                # Save to file
                output_filename = "reconstructed_beatmap.osu"
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(osu_output_string)
                print(f"Saved reconstructed beatmap to: {output_filename}")
            else:
                print("\nFailed to generate .osu string.")

        except FileNotFoundError as e:
            print(f"\nError during test: {e}")
            print("Ensure Node.js is installed and the node encoder script path is correct.")
        except Exception as e:
            print(f"\nAn error occurred during testing: {e}")
    else:
         print(f"\nSkipping Node.js encoding test: Encoder script not found at {encoder_script_full_path}")