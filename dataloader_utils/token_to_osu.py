import json
import subprocess
from pathlib import Path
import sys
import math
import traceback

from quantizers import Quantizers
from vocabulary import BeatmapVocabulary
  

class TokenToOsu:
    """
    Converts a sequence of beatmap tokens back into an osu! file format string
    by reconstructing hit objects and control points, then using a Node.js encoder script.
    Handles different slider types (L, B, P, C) and timing points (excluding Kiai).
    """
    def __init__(self, quantizers, vocab, node_encoder_script="encoder/encode_osu.js"):
        """
        Initializes the detokenizer.

        Args:
            quantizers (QuantizationManager): Instance with dequantize methods.
            vocab (BeatmapVocabulary): Instance for token mapping.
            node_encoder_script (str): Relative path to the Node.js encoder script
                                       (from the project root or script location).
        """
        self.quantizers = quantizers
        self.vocab = vocab

        # Determine the absolute path to the node script robustly
        script_dir = Path(__file__).parent.parent # Assumes this script is in dataloader_utils
        self.node_encoder_path = (script_dir / node_encoder_script).resolve()
        if not self.node_encoder_path.is_file():
             # Try path relative to current working directory as fallback
             self.node_encoder_path = Path(node_encoder_script).resolve()
             if not self.node_encoder_path.is_file():
                 raise FileNotFoundError(f"Node encoder script not found at expected locations: {script_dir / node_encoder_script} or {Path(node_encoder_script).resolve()}")
        print(f"DEBUG: Using Node encoder script at: {self.node_encoder_path}")

        # Internal state during detokenization - reset before each use
        self._reset_state()

    def _reset_state(self):
        """Resets the internal state for processing a new sequence."""
        self.hit_objects = []
        self.control_points = []
        # Track current timing state for reconstruction (used for defaults if tokens missing)
        self.current_beat_length = 500.0 # Default 120 BPM
        self.current_sv_multiplier = 1.0
        self.current_time_signature = 4
        # --- End Timing State ---
        self.current_time_ms = 0.0 # Use float for potentially fractional time shifts
        self.pending_object_type = None
        self.pending_x_bin = None
        self.pending_y_bin = None
        self.new_combo_pending = False
        # Stores {'type': 'L'/'B'/'P'/'C', 'start_time': ms, 'points': [{'x':px, 'y':px}, ...], 'repeats': n, 'is_new_combo': bool}
        self.current_slider_data = None
        self.spinner_start_time = 0.0
        # State for pending timing point
        self.pending_timing_point_type = None # 'Uninherited' or 'Inherited'
        self.pending_timing_data = {} # Store values for the current timing point

    def _finalize_pending_timing_point(self):
        """Adds the currently pending timing point data to the control_points list."""
        if self.pending_timing_point_type:
             # Add time to the data
             self.pending_timing_data['time'] = round(self.current_time_ms)
             self.pending_timing_data['pointType'] = self.pending_timing_point_type

             # Add default/last known values if not set by tokens for this specific point
             # These are needed because osu-classes requires a full set of points (Timing, Diff, Sample, Effect)
             # at each timing change, even if only one aspect (like SV) changed in the tokens.
             if self.pending_timing_point_type == 'Uninherited':
                  self.pending_timing_data.setdefault('beatLength', self.current_beat_length)
                  self.pending_timing_data.setdefault('timeSignature', self.current_time_signature)
             # SV applies to both types implicitly if not overridden
             self.pending_timing_data.setdefault('sliderVelocity', self.current_sv_multiplier)
             # print(f"DEBUG: Finalizing Timing Point: {self.pending_timing_data}") # Debug
             self.control_points.append(self.pending_timing_data)

        # Reset pending state
        self.pending_timing_point_type = None
        self.pending_timing_data = {}


    def detokenize(self, token_id_sequence):
        """
        Converts a sequence of token IDs into lists of hit object and control point dictionaries.

        Args:
            token_id_sequence (list[int]): The sequence of token IDs.

        Returns:
            tuple(list[dict], list[dict]): A tuple containing:
                - A list of hit object dictionaries.
                - A list of control point dictionaries.
                Both suitable for the Node.js encoder. Returns empty lists if input is invalid.
        """
        self._reset_state()

        # --- Input Validation and Cleaning ---
        if not isinstance(token_id_sequence, (list, tuple)):
             print("Error: Input token_id_sequence must be a list or tuple.")
             return [], []
        if not token_id_sequence:
             print("Warning: Input token sequence is empty.")
             return [], []
        if token_id_sequence[0] != self.vocab.sos_id:
             print("Warning: Input sequence does not start with SOS token.")
             # Decide whether to proceed or return error

        # Find end of sequence (EOS or first PAD)
        end_index = len(token_id_sequence)
        try: end_index = token_id_sequence.index(self.vocab.eos_id)
        except ValueError:
            try: end_index = token_id_sequence.index(self.vocab.pad_id)
            except ValueError: pass
        processed_sequence = token_id_sequence[1:end_index]
        if not processed_sequence: print("Warning: Empty sequence after removing SOS/EOS/PAD."); return [], []

        # --- Token Iteration ---
        for i, token_id in enumerate(processed_sequence):
            token_str = self.vocab.get_token(token_id)
            if token_str is None: print(f"Warning: Unknown token ID {token_id} at index {i+1}. Skipping."); continue

            # --- Handle Time Shift FIRST ---
            if token_str.startswith("TIME_SHIFT_"):
                 self._finalize_pending_timing_point() # Finalize timing point *before* advancing time
                 try:
                    time_bin = int(token_str.split('_')[-1])
                    dt_ms = self.quantizers.dequantize_time_shift(time_bin)
                    self.current_time_ms += dt_ms
                 except Exception as e: print(f"Error processing time token '{token_str}': {e}")
                 continue # Time shift processed, move to next token

            # --- Handle Timing Point Markers ---
            elif token_str == "UNINHERITED_TIMING_POINT":
                 self._finalize_pending_timing_point() # Finalize previous if any
                 self.pending_timing_point_type = 'Uninherited'
                 self.pending_timing_data = {} # Start fresh data for this point
            elif token_str == "INHERITED_TIMING_POINT":
                 self._finalize_pending_timing_point() # Finalize previous if any
                 self.pending_timing_point_type = 'Inherited'
                 self.pending_timing_data = {} # Start fresh data for this point

            # --- Handle Timing Point Value Tokens ---
            elif token_str.startswith("BEAT_LENGTH_BIN_"):
                 if self.pending_timing_point_type:
                     try:
                         bl_bin = int(token_str.split('_')[-1])
                         bl_ms = self.quantizers.dequantize_beat_length(bl_bin)
                         self.pending_timing_data['beatLength'] = bl_ms
                         self.current_beat_length = bl_ms # Update overall state
                     except Exception as e: 
                         print(f"Error processing beat length token '{token_str}': {e}")
                 else: 
                     print(f"Warning: Beat length token '{token_str}' found without pending timing point.")
            elif token_str.startswith("SLIDER_VELOCITY_BIN_"):
                 if self.pending_timing_point_type:
                     try:
                         sv_bin = int(token_str.split('_')[-1])
                         sv_mult = self.quantizers.dequantize_slider_velocity(sv_bin)
                         self.pending_timing_data['sliderVelocity'] = sv_mult
                         self.current_sv_multiplier = sv_mult # Update overall state
                     except Exception as e: 
                         print(f"Error processing SV token '{token_str}': {e}")
                 else: 
                     print(f"Warning: SV token '{token_str}' found without pending timing point.")
            elif token_str.startswith("TIME_SIGNATURE_"):
                 if self.pending_timing_point_type == 'Uninherited': # Only applies to uninherited
                     try:
                         ts = int(token_str.split('_')[-1])
                         self.pending_timing_data['timeSignature'] = ts
                         self.current_time_signature = ts # Update overall state
                     except Exception as e: 
                         print(f"Error processing TS token '{token_str}': {e}")
                 else: 
                     print(f"Warning: Time signature token '{token_str}' found without pending uninherited timing point.")

            # --- Handle Hit Object Tokens (after finalizing any pending timing point) ---
            elif token_str == "NEW_COMBO":
                 self._finalize_pending_timing_point()
                 self.new_combo_pending = True
            elif token_str == "PLACE_CIRCLE":
                 self._finalize_pending_timing_point()
                 self._finalize_previous_object() # Finalize pending slider etc.
                 self.pending_object_type = 'circle'
            elif token_str.startswith("START_SLIDER_"): # Handles L, B, P, C
                 self._finalize_pending_timing_point()
                 self._finalize_previous_object()
                 self.pending_object_type = 'slider_start'
                 slider_type_char = token_str[-1] # L, B, P, or C
                 self.current_slider_data = {
                     'type': slider_type_char,
                     'start_time': self.current_time_ms,
                     'points': [],
                     'repeats': 0,
                     'is_new_combo': self.new_combo_pending
                 }
                 self.new_combo_pending = False # Consume flag
            elif token_str == "ADD_SLIDER_ANCHOR":
                 # Anchor belongs to the current slider, don't finalize timing point
                 if self.current_slider_data and self.current_slider_data['points']:
                      self.pending_object_type = 'slider_anchor'
                 else:
                     print(f"Warning: ADD_SLIDER_ANCHOR token ignored at {self.current_time_ms:.0f} (no active slider/start point).")
                     self.pending_object_type = None
            elif token_str == "PLACE_SPINNER":
                 self._finalize_pending_timing_point()
                 self._finalize_previous_object()
                 self.pending_object_type = 'spinner'
                 self.spinner_start_time = self.current_time_ms

            # --- Handle Coordinate Tokens ---
            elif token_str.startswith("COORD_X_"):
                 # Coords belong to pending object, don't finalize timing point
                 try: 
                     self.pending_x_bin = int(token_str.split('_')[-1])
                     self._try_process_coordinates()
                 except Exception as e:
                    print(f"Error parsing X coord token '{token_str}': {e}")
                    self.pending_x_bin = None
            elif token_str.startswith("COORD_Y_"):
                 # Coords belong to pending object, don't finalize timing point
                 try: 
                    self.pending_y_bin = int(token_str.split('_')[-1])
                    self._try_process_coordinates()
                 except Exception as e:
                    print(f"Error parsing Y coord token '{token_str}': {e}")
                    self.pending_y_bin = None

            # --- Handle Object End Tokens ---
            elif token_str.startswith("END_SLIDER_"):
                 # End token belongs to slider, don't finalize timing point
                 if self.current_slider_data:
                     try:
                         repeat_bin = int(token_str.split('_')[-1][:-1])
                         self.current_slider_data['repeats'] = self.quantizers.dequantize_slider_repeats(repeat_bin)
                         self._finalize_slider() # Add slider to hit_objects list
                     except Exception as e:
                         print(f"Error processing slider end token '{token_str}': {e}")
                         self.current_slider_data = None # Discard incomplete slider
                 else: print(f"Warning: END_SLIDER token ignored at {self.current_time_ms:.0f} (no active slider).")
                 self.pending_object_type = None
                 self.pending_x_bin = None
                 self.pending_y_bin = None # Reset state
            elif token_str.startswith("END_SPINNER_DUR"):
                 # End token belongs to spinner, don't finalize timing point
                 if self.pending_object_type == 'spinner':
                     try:
                         duration_bin = int(token_str[len("END_SPINNER_DUR"):])
                         duration_ms = self.quantizers.dequantize_spinner_duration(duration_bin)
                         spinner_obj = {
                             'object_type': 'Spinner',
                             'time': round(self.spinner_start_time),
                             'end_time': round(self.spinner_start_time + duration_ms),
                             'x': 256,
                             'y': 192, # Fixed position
                             'is_new_combo': self.new_combo_pending
                         }
                         self.hit_objects.append(spinner_obj)
                         self.new_combo_pending = False
                         self.pending_object_type = None # Reset state
                     except Exception as e:
                         print(f"Error processing spinner end token '{token_str}': {e}")
                         self.pending_object_type = None
                 else:
                     print(f"Warning: END_SPINNER token ignored at {self.current_time_ms:.0f} (no active spinner).")
                     self.pending_object_type = None # Reset state

        # --- Finalization after Loop ---
        self._finalize_pending_timing_point() # Finalize any timing point pending at the end
        self._finalize_previous_object()    # Finalize any hit object pending at the end

        print(f"Detokenization complete. Reconstructed {len(self.hit_objects)} hit objects and {len(self.control_points)} control points.")
        print(self.hit_objects)
        return self.hit_objects, self.control_points # Return both lists

    def _try_process_coordinates(self):
        """Checks if both X and Y bins are ready and calls processing if so."""
        if self.pending_x_bin is not None and self.pending_y_bin is not None:
            self._process_coordinates() # Process the pair

    def _process_coordinates(self):
        """Processes pending coordinate bins based on the pending object type."""
        if self.pending_x_bin is None or self.pending_y_bin is None: return # Safety check
        try:
            if self.pending_object_type == 'circle':
                abs_x = self.quantizers.dequantize_coord(self.pending_x_bin, 'x')
                abs_y = self.quantizers.dequantize_coord(self.pending_y_bin, 'y')
                circle_obj = {
                    'object_type': 'Circle', 
                    'time': round(self.current_time_ms), 
                    'x': abs_x, 
                    'y': abs_y, 
                    'is_new_combo': self.new_combo_pending
                }
                self.hit_objects.append(circle_obj)
                self.new_combo_pending = False
            elif self.pending_object_type == 'slider_start':
                 if self.current_slider_data:
                    abs_x = self.quantizers.dequantize_coord(self.pending_x_bin, 'x')
                    abs_y = self.quantizers.dequantize_coord(self.pending_y_bin, 'y')
                    self.current_slider_data['points'].append({'x': abs_x, 'y': abs_y})
                 else: print("Error: Slider start coords without slider data.")
            elif self.pending_object_type == 'slider_anchor':
                if self.current_slider_data and self.current_slider_data['points']:
                    dx = self.quantizers.dequantize_relative_coord(self.pending_x_bin, 'x')
                    dy = self.quantizers.dequantize_relative_coord(self.pending_y_bin, 'y')
                    last_point = self.current_slider_data['points'][-1]
                    new_x = round(last_point['x'] + dx)
                    new_y = round(last_point['y'] + dy)
                    self.current_slider_data['points'].append({'x': new_x, 'y': new_y})
                else: print("Error: Slider anchor coords without valid slider data.")
            # Reset pending state after processing coordinates
            self.pending_object_type = None # Coords processed, wait for next action/end
            self.pending_x_bin = None
            self.pending_y_bin = None
        except Exception as e:
             print(f"Error during coordinate processing: {e}"); traceback.print_exc()
             self.pending_object_type = None; self.pending_x_bin = None; self.pending_y_bin = None # Reset state

    def _finalize_slider(self):
        """Adds the completed slider data to the hit objects list if valid."""
        if self.current_slider_data:
            if len(self.current_slider_data['points']) >= 2:
                start_point = self.current_slider_data['points'][0]
                slider_obj = {
                    'object_type': 'Slider',
                    'time': round(self.current_slider_data['start_time']),
                    'x': start_point['x'], 
                    'y': start_point['y'],
                    'is_new_combo': self.current_slider_data['is_new_combo'],
                    'slider_type_char': self.current_slider_data['type'], # L, B, P, C
                    'repeats': self.current_slider_data['repeats'],
                    'points': self.current_slider_data['points'] # List of {x,y} dicts
                }
                self.hit_objects.append(slider_obj)
            else: print(f"Warning: Discarding slider at {self.current_slider_data['start_time']:.0f} (insufficient points).")
        self.current_slider_data = None

    def _finalize_previous_object(self):
        """Finalizes any pending slider."""
        if self.current_slider_data:
            print(f"Warning: Finalizing potentially incomplete slider at {self.current_slider_data['start_time']:.0f}.")
            self._finalize_slider()

    def get_osu_string(self, token_id_sequence, metadata_overrides=None, difficulty_overrides=None):
        """
        Detokenizes the sequence including control points and calls the Node.js encoder.

        Args:
            token_id_sequence (list[int]): The sequence of token IDs.
            metadata_overrides (dict, optional): Dictionary to override default metadata.
            difficulty_overrides (dict, optional): Dictionary to override default difficulty.

        Returns:
            str: The generated .osu file content as a string, or None on error.
        """
        # Step 1: Convert token sequence to lists of hit objects and control points
        hit_object_list, control_point_list = self.detokenize(token_id_sequence)
        
        if not hit_object_list and not control_point_list:
             print("Warning: No hit objects or control points reconstructed. Cannot generate .osu string.")
             return None

        # Step 2: Call the Node.js encoder with the reconstructed data
        encoder_result = self._call_node_encoder(hit_object_list, control_point_list, metadata_overrides, difficulty_overrides)
        encoder_result = encoder_result[encoder_result.find("osu file format v"):]
        return encoder_result
    
    def _call_node_encoder(self, hit_object_list, control_point_list, metadata_overrides, difficulty_overrides):
        """
        Helper method to prepare JSON payload (including control points)
        and run the Node.js encoder subprocess.
        """
        payload = {
            'metadata': metadata_overrides or {},
            'difficulty': difficulty_overrides or {},
            'control_points': control_point_list, # Pass reconstructed points
            'hit_objects': hit_object_list
        }
        try: payload_json = json.dumps(payload)
        except TypeError as e: print(f"Error serializing payload to JSON: {e}"); return None

        try:
            command = ['node', str(self.node_encoder_path)]
            cwd_path = self.node_encoder_path.parent
            print(f"DEBUG: Calling Node encoder: {' '.join(command)}")

            result = subprocess.run(command, input=payload_json, capture_output=True, text=True, encoding='utf-8', check=True, cwd=cwd_path, timeout=30)
            osu_string = result.stdout
            #if not osu_string or not osu_string.startswith("osu file format v"):
            #     print(f"Error: Node encoder invalid output. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"); return None
            #print("DEBUG: Successfully received .osu string from Node encoder.")
            return osu_string
        except FileNotFoundError: print("Error: 'node' command not found."); return None
        except subprocess.CalledProcessError as e: print(f"Error: Node encoder failed (Code {e.returncode}). stderr:\n{e.stderr}"); return None
        except subprocess.TimeoutExpired: print("Error: Node encoder timed out."); return None
        except Exception as e: print(f"Unexpected error calling Node encoder: {e}"); traceback.print_exc(); return None

    def save_osu_string(self, osu_string, filename):
        """Saves the .osu string to a file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(osu_string)
            print(f"DEBUG: Saved .osu string to {filename}")
        except IOError as e:
            print(f"Error saving .osu string to {filename}: {e}")
            return False
        return True