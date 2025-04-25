# dataloader_utils/osu_to_token.py
import json
import subprocess
from pathlib import Path
import sys
import traceback
from operator import itemgetter # For sorting events

from quantizers import Quantizers
from vocabulary import BeatmapVocabulary


class OsuToToken:
    def __init__(self, osu_file_path, quantizers, vocab, max_seq_len=20480, node_parser_script="parser/parse_osu.js"):
        """
        Initializes the tokenizer, now including timing point processing.
        """
        self.osu_file_path = osu_file_path
        self.quantizers = quantizers
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        # Resolve node parser path
        script_dir = Path(__file__).parent.parent
        self.node_parser_path = (script_dir / node_parser_script).resolve()
        if not self.node_parser_path.is_file():
             self.node_parser_path = Path(node_parser_script).resolve()
             if not self.node_parser_path.is_file():
                 raise FileNotFoundError(f"Node parser script not found at expected locations: {script_dir / node_parser_script} or {Path(node_parser_script).resolve()}")

        # Run parser and store data
        self.parsed_data = self._run_node_parser()

        # Initialize token sequence
        self.token_id_sequence = []
        self.token_id_sequence.append(self.vocab.sos_id)

        # Tokenize if parser succeeded
        if self.parsed_data and ('hitObjects' in self.parsed_data or 'controlPoints' in self.parsed_data):
             self.tokenize_osu()
        else:
             print(f"Warning: No hit objects or control points found, or error parsing file: {self.osu_file_path}")
             if self.vocab: self._finalize_and_pad()


    def _run_node_parser(self):
        """Executes the Node.js parser script and returns the parsed JSON data."""
        # (Implementation remains the same as in tokenizer_timing artifact)
        osu_path_str = None
        try:
            if isinstance(self.osu_file_path, (list, tuple)) and len(self.osu_file_path) > 0: osu_path_str = self.osu_file_path[0]
            elif isinstance(self.osu_file_path, (str, Path)): osu_path_str = self.osu_file_path
            if not isinstance(osu_path_str, (str, Path)): print(f"Error: Invalid osu_file_path type: {type(osu_path_str)}"); return None
            osu_abs_path = str(Path(osu_path_str).resolve())
            node_script_abs_path = str(self.node_parser_path.resolve())
            cwd_path = self.node_parser_path.parent
            if not Path(osu_abs_path).is_file(): print(f"Error: osu file not found: {osu_abs_path}"); return None
            if not Path(node_script_abs_path).is_file(): print(f"Error: Node script not found: {node_script_abs_path}"); return None
        except Exception as e: print(f"Error processing paths: {e}"); traceback.print_exc(); return None

        try:
            command = ['node', node_script_abs_path, osu_abs_path]
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True, cwd=cwd_path, timeout=30)
            stdout_trimmed = result.stdout.strip()
            if not stdout_trimmed or not stdout_trimmed.startswith('{'): print(f"Error: Node output invalid/empty. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"); return None
            return json.loads(stdout_trimmed)
        except FileNotFoundError: print("Error: 'node' command not found."); return None
        except subprocess.CalledProcessError as e: print(f"Error: Node parser failed (Code {e.returncode}). stderr:\n{e.stderr}"); return None
        except subprocess.TimeoutExpired: print("Error: Node parser timed out."); return None
        except json.JSONDecodeError as e: print(f"Error decoding JSON from Node: {e}. stdout:\n{result.stdout}"); return None
        except Exception as e: print(f"Unexpected error running Node parser: {e}"); traceback.print_exc(); return None
        
    def _append_token(self, token_str):
        """Appends a token ID to the sequence, checking length limits."""
        if len(self.token_id_sequence) >= self.max_seq_len - 1: return False # Full
        token_id = self.vocab.get_id(token_str)
        if token_id is None: print(f"Warning: Token not found in vocab: '{token_str}'. Skipping."); return True # Skip token but continue sequence
        self.token_id_sequence.append(token_id)
        return True

    def _finalize_and_pad(self):
        """Handles EOS token appending, truncation, and padding."""
        # Ensure sequence starts with SOS if somehow lost
        if not self.token_id_sequence or self.token_id_sequence[0] != self.vocab.sos_id:
             self.token_id_sequence.insert(0, self.vocab.sos_id)

        # If sequence reached max length *before* adding EOS
        if len(self.token_id_sequence) >= self.max_seq_len:
            self.token_id_sequence = self.token_id_sequence[:self.max_seq_len]
            if self.token_id_sequence[-1] != self.vocab.eos_id:
                 print(f"Warning: Sequence truncated at {self.max_seq_len} tokens, forcing EOS for {self.osu_file_path}.")
                 self.token_id_sequence[-1] = self.vocab.eos_id
        # If sequence is shorter and doesn't end with EOS
        elif not self.token_id_sequence or self.token_id_sequence[-1] != self.vocab.eos_id:
            self.token_id_sequence.append(self.vocab.eos_id)

        # Pad
        padding_needed = self.max_seq_len - len(self.token_id_sequence)
        if padding_needed > 0:
            self.token_id_sequence.extend([self.vocab.pad_id] * padding_needed)

        # Final check
        self.token_id_sequence = self.token_id_sequence[:self.max_seq_len]


    def tokenize_osu(self):
        """Tokenizes parsed osu data, including timing points (excluding Kiai)."""
        last_event_time_ms = 0.0
        sequence_full = False
        events = []

        # --- Prepare Combined Event List ---
        for ho in self.parsed_data.get('hitObjects', []):
            events.append({'time': ho['startTime'], 'type': 'hit_object', 'data': ho})

        # Add control points - simplified significance check (only Uninherited/Inherited)
        for cp in self.parsed_data.get('controlPoints', []):
            cp_type = cp.get('pointType', 'Unknown')
            # Only consider points that change BPM/SV or Time Signature as significant for tokenization now
            if cp_type in ['Uninherited', 'Inherited']:
                events.append({'time': cp['time'], 'type': 'control_point', 'data': cp})
            # Removed check for Kiai change on EffectOnly/SampleOnly points

        # Sort all events chronologically. Control points first at same time.
        events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'control_point' else 1))

        # --- Iterate Through Sorted Events ---
        # Removed current_kiai_state tracking

        for event in events:
            if sequence_full: break

            current_event_time_ms = event['time']
            time_difference_ms = max(0.0, current_event_time_ms - last_event_time_ms)

            # --- 1. Time Shift ---
            try:
                time_bin = self.quantizers.quantize_time_shift(time_difference_ms)
                if not self._append_token(f"TIME_SHIFT_{time_bin}"): sequence_full = True; continue
            except AttributeError: print("Error: Quantizer missing 'quantize_time_shift'."); sequence_full=True; continue
            except Exception as e: print(f"Error quantizing time shift {time_difference_ms}: {e}"); sequence_full=True; continue

            # --- 2. Process Event Data ---
            event_type = event['type']
            event_data = event['data']

            # --- 2a. Control Point ---
            if event_type == 'control_point':
                cp_type = event_data['pointType']

                try:
                    if cp_type == 'Uninherited':
                        if not self._append_token("UNINHERITED_TIMING_POINT"): sequence_full = True; break
                        bl_ms = event_data.get('beatLength')
                        if bl_ms is not None and bl_ms > 0:
                             bl_bin = self.quantizers.quantize_beat_length(bl_ms)
                             if not self._append_token(f"BEAT_LENGTH_BIN_{bl_bin}"): sequence_full = True; break
                        else: print(f"Warning: Invalid beatLength {bl_ms} at time {current_event_time_ms}. Skipping token.")
                        ts = event_data.get('timeSignature', 4)
                        if ts in self.vocab.supported_time_signatures:
                             if not self._append_token(f"TIME_SIGNATURE_{ts}"): sequence_full = True; break

                    elif cp_type == 'Inherited':
                        if not self._append_token("INHERITED_TIMING_POINT"): sequence_full = True; break
                        sv = event_data.get('sliderVelocity', 1.0)
                        sv_bin = self.quantizers.quantize_slider_velocity(sv)
                        if not self._append_token(f"SLIDER_VELOCITY_BIN_{sv_bin}"): sequence_full = True; break

                except AttributeError as e: print(f"Error: Quantizer missing method for timing point: {e}"); sequence_full=True; break
                except Exception as e: print(f"Error processing control point at {current_event_time_ms}: {e}"); sequence_full=True; break

            # --- 2b. Hit Object ---
            elif event_type == 'hit_object':
                # (Hit object logic remains exactly the same as in tokenizer_timing artifact)
                ho_data = event_data
                try:
                    if ho_data.get('isNewCombo', False):
                         if not self._append_token("NEW_COMBO"): sequence_full = True; continue
                    
                    object_type = ho_data['objectType']
                    
                    if object_type == 'Circle':
                        if not self._append_token("PLACE_CIRCLE"): sequence_full = True; continue
                        
                        x_bin = self.quantizers.quantize_coord(ho_data['startX'], 'x')
                        y_bin = self.quantizers.quantize_coord(ho_data['startY'], 'y')
                        if not self._append_token(f"COORD_X_{x_bin}"): sequence_full = True; continue
                        if not self._append_token(f"COORD_Y_{y_bin}"): sequence_full = True; continue
                    
                    elif object_type == 'Slider':
                        slider_path = ho_data.get('path')
                        if not slider_path: 
                            print(f"Warning: Slider at {current_event_time_ms} missing path data.")
                            continue

                        curve_type_char = slider_path.get('curveTypeChar', 'B')
                        start_token = f"START_SLIDER_{curve_type_char}"

                        if not self._append_token(start_token): sequence_full = True; continue
                        
                        start_x_bin = self.quantizers.quantize_coord(ho_data['startX'], 'x')
                        start_y_bin = self.quantizers.quantize_coord(ho_data['startY'], 'y')
                        if not self._append_token(f"COORD_X_{start_x_bin}"): sequence_full = True; continue
                        if not self._append_token(f"COORD_Y_{start_y_bin}"): sequence_full = True; continue

                        control_points = slider_path.get('controlPoints', [])
                        last_point_abs = {
                            'x': ho_data['startX'], 
                            'y': ho_data['startY']
                        }
                        #for i in range(1, len(control_points)):
                        for anchor_point_rel in control_points[1:]:
                            anchor_point_abs = {
                                'x': ho_data['startX'] + anchor_point_rel['x'], 
                                'y': ho_data['startY'] + anchor_point_rel['y']
                            }
                            dx = anchor_point_abs['x'] - last_point_abs['x']
                            dy = anchor_point_abs['y'] - last_point_abs['y']

                            dx_bin = self.quantizers.quantize_relative_coord(dx, 'x')
                            dy_bin = self.quantizers.quantize_relative_coord(dy, 'y')

                            if not self._append_token("ADD_SLIDER_ANCHOR"): sequence_full = True; break
                            if not self._append_token(f"COORD_X_{dx_bin}"): sequence_full = True; break
                            if not self._append_token(f"COORD_Y_{dy_bin}"): sequence_full = True; break
                            last_point_abs = anchor_point_abs
                        
                        if sequence_full: 
                            continue
                        
                        repeats = ho_data.get('repeats', 0)
                        repeat_bin = self.quantizers.quantize_slider_repeats(repeats)
                        if not self._append_token(f"END_SLIDER_{repeat_bin}R"): sequence_full = True; continue
                    
                    elif object_type == 'Spinner':
                        if not self._append_token("PLACE_SPINNER"): sequence_full = True; continue

                        end_time = ho_data.get('endTime', current_event_time_ms)
                        duration_ms = max(0.0, end_time - current_event_time_ms)
                        duration_bin = self.quantizers.quantize_spinner_duration(duration_ms)
                        
                        if not self._append_token(f"END_SPINNER_DUR{duration_bin}"): sequence_full = True; continue
                    
                    else: 
                        print(f"Warning: Unknown object type '{object_type}' at {current_event_time_ms}.")
                
                except AttributeError as e: 
                    print(f"Error: Quantizer missing method for hit object: {e}")
                    sequence_full=True
                    continue
                except Exception as e: 
                    print(f"Error processing hit object at {current_event_time_ms}: {e}")
                    sequence_full=True
                    continue

            last_event_time_ms = current_event_time_ms

        # --- Finalize Sequence (EOS, Padding/Truncation) ---
        self._finalize_and_pad()
        print(f"Tokenization complete for {self.osu_file_path}. Sequence length: {len(self.token_id_sequence)}")
        return self.token_id_sequence

    def get_token_sequence(self):
        """Returns the final token ID sequence."""
        return self.token_id_sequence
