import math
import json 
from dataloader_utils.quantizers import Quantizers

class BeatmapVocabulary:
    def __init__(self, time_shift_bins=64, coord_x_bins=32, coord_y_bins=24, max_slider_repeats=4, spinner_duration_bins=16):
        self.time_shift_bins = time_shift_bins
        self.coord_x_bins = coord_x_bins
        self.coord_y_bins = coord_y_bins
        self.max_slider_repeats = max_slider_repeats
        self.spinner_duration_bins = spinner_duration_bins
        
        # Initialize dictionaries for mapping
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0

        self._build_vocabulary()

    def _add_token(self, token_str):
        """Adds a token to the vocabulary if it's not already present."""
        if token_str not in self.token_to_id:
            new_id = self.vocab_size
            self.token_to_id[token_str] = new_id
            self.id_to_token[new_id] = token_str
            self.vocab_size += 1
        return self.token_to_id[token_str]

    def _build_vocabulary(self):
        """Systematically adds all defined tokens."""
        print("Building Vocabulary...")

        # 1. Special Tokens (Essential - assign IDs 0, 1, 2 first)
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_id = self._add_token(self.pad_token) # Should be 0
        self.sos_id = self._add_token(self.sos_token) # Should be 1
        self.eos_id = self._add_token(self.eos_token) # Should be 2
        print(f"Added Special Tokens (PAD={self.pad_id}, SOS={self.sos_id}, EOS={self.eos_id})")

        # 2. Time Shift Tokens
        print(f"Adding {self.time_shift_bins} Time Shift Tokens...")
        for i in range(self.time_shift_bins):
            self._add_token(f"TIME_SHIFT_{i}")

        # 3. Coordinate Tokens
        print(f"Adding {self.coord_x_bins} X Coordinate Tokens...")
        for i in range(self.coord_x_bins):
            self._add_token(f"COORD_X_{i}")
        print(f"Adding {self.coord_y_bins} Y Coordinate Tokens...")
        for i in range(self.coord_y_bins):
            self._add_token(f"COORD_Y_{i}")

        # 4. Action / Type Tokens
        print("Adding Action/Type Tokens...")
        action_tokens = [
            "NEW_COMBO",
            "PLACE_CIRCLE",
            "START_SLIDER_L", # Linear
            "START_SLIDER_B", # Bezier
            "START_SLIDER_P", # Perfect Circle
            "ADD_SLIDER_ANCHOR",
            "PLACE_SPINNER",
        ]
        for token in action_tokens:
            self._add_token(token)

        # 5. Slider End Tokens (with repeats)
        print(f"Adding Slider End Tokens (0 to {self.max_slider_repeats} repeats)...")
        for i in range(self.max_slider_repeats + 1):
             self._add_token(f"END_SLIDER_{i}R")

        # 6. Spinner End Tokens (with duration bins)
        print(f"Adding Spinner End Tokens ({self.spinner_duration_bins} duration bins)...")
        for i in range(self.spinner_duration_bins):
             self._add_token(f"END_SPINNER_DUR{i}")

        print(f"--- Vocabulary Built! Total size: {self.vocab_size} ---")

    def get_id(self, token_str):
        """Gets the ID for a token string. Handles unknown tokens if needed."""
        # Option 1: Return a special UNK token ID (if you add one)
        # Option 2: Raise an error
        # Option 3: Return None
        return self.token_to_id.get(token_str) # Returns None if not found

    def get_token(self, token_id):
        """Gets the token string for an ID."""
        return self.id_to_token.get(token_id) # Returns None if not found

    def __len__(self):
        """Returns the total size of the vocabulary."""
        return self.vocab_size

    def save(self, filepath="vocabulary.json"):
        """Saves the vocabulary mappings to a JSON file."""
        # We only need to save token_to_id, the rest can be reconstructed
        data = {
            'token_to_id': self.token_to_id,
            'time_shift_bins': self.time_shift_bins,
            'coord_bins': self.coord_bins,
            'max_slider_repeats': self.max_slider_repeats,
            'spinner_duration_bins': self.spinner_duration_bins,
            'pad_token': self.pad_token,
            'sos_token': self.sos_token,
            'eos_token': self.eos_token
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Vocabulary saved to {filepath}")
        except IOError as e:
            print(f"Error saving vocabulary: {e}")

    @classmethod
    def load(cls, filepath="vocabulary.json"):
        """Loads the vocabulary from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Create a new instance without calling build_vocabulary initially
            instance = cls.__new__(cls) # Create instance without calling __init__

            # Restore state from saved data
            instance.token_to_id = data['token_to_id']
            instance.vocab_size = len(instance.token_to_id)
            instance.id_to_token = {v: k for k, v in instance.token_to_id.items()} # Rebuild id_to_token

            # Restore parameters
            instance.time_shift_bins = data.get('time_shift_bins', 64) 
            instance.coord_bins = data.get('coord_bins', 32)
            instance.max_slider_repeats = data.get('max_slider_repeats', 4)
            instance.spinner_duration_bins = data.get('spinner_duration_bins', 16)
            instance.pad_token = data.get('pad_token', '<PAD>')
            instance.sos_token = data.get('sos_token', '<SOS>')
            instance.eos_token = data.get('eos_token', '<EOS>')

            # Ensure special IDs are consistent
            instance.pad_id = instance.token_to_id.get(instance.pad_token, 0)
            instance.sos_id = instance.token_to_id.get(instance.sos_token, 1)
            instance.eos_id = instance.token_to_id.get(instance.eos_token, 2)


            print(f"Vocabulary loaded from {filepath}, size: {instance.vocab_size}")
            return instance
        except FileNotFoundError:
            print(f"Error: Vocabulary file not found at {filepath}. Create a new one.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filepath}.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred loading vocabulary: {e}")
            return None


if __name__ == "__main__":
    # Create a new vocabulary
    vocab = BeatmapVocabulary()

    # Print some info
    print(f"\nVocabulary Size: {len(vocab)}")
    print(f"ID for '<PAD>': {vocab.pad_id}")
    print(f"ID for 'PLACE_CIRCLE': {vocab.get_id('PLACE_CIRCLE')}")
    print(f"ID for 'COORD_X_15': {vocab.get_id('COORD_X_15')}")
    print(f"Token for ID 50: {vocab.get_token(50)}")

    # Save the vocabulary
    vocab.save("my_beatmap_vocab.json")

    # Load the vocabulary (demonstration)
    loaded_vocab = BeatmapVocabulary.load("my_beatmap_vocab.json")
    if loaded_vocab:
        print(f"\nLoaded Vocabulary Size: {len(loaded_vocab)}")
        print(f"Loaded ID for 'PLACE_CIRCLE': {loaded_vocab.get_id('PLACE_CIRCLE')}")
        print(f"Token for ID {loaded_vocab.pad_id}: {loaded_vocab.get_token(loaded_vocab.pad_id)}")

    # --- Quantization Function Placeholders (Need actual implementation in Tokenizer) ---
    def get_time_bin(dt_ms, num_bins=64):
        # Placeholder: Implement logarithmic or other scaling logic here
        # Example: very basic linear scaling (replace with better logic)
        max_dt = 3000 # Assume max time shift considered is 3000ms
        bin_index = int((dt_ms / max_dt) * num_bins)
        return min(bin_index, num_bins - 1) # Clamp to max bin index

    def get_coord_bin(coord_norm, num_bins=32):
        # coord_norm is assumed to be between 0.0 and 1.0
        if not (0.0 <= coord_norm <= 1.0):
             # Handle edge case slightly outside 0-1 due to float precision
             coord_norm = max(0.0, min(1.0, coord_norm))
        # Ensure index doesn't accidentally become num_bins due to coord_norm being exactly 1.0
        bin_index = min(int(coord_norm * num_bins), num_bins - 1)
        return bin_index

    # Example usage of quantization placeholders:
    time_token = f"TIME_SHIFT_{get_time_bin(125.5)}" # Calculate time shift between objects
    x_coord_token = f"COORD_X_{get_coord_bin(256 / 512.0)}" # Normalize X first
    y_coord_token = f"COORD_Y_{get_coord_bin(192 / 384.0)}" # Normalize Y first
    print(f"\nExample quantization:")
    print(f"Time shift 125.5ms -> {time_token} -> ID: {vocab.get_id(time_token)}")
    print(f"Coords (256, 192) -> {x_coord_token}, {y_coord_token} -> IDs: {vocab.get_id(x_coord_token)}, {vocab.get_id(y_coord_token)}")
