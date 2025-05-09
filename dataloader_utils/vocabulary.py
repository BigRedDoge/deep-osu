import math
import json

class BeatmapVocabulary:
    # Remove slider_duration_bins from init parameters and storage
    def __init__(self, 
                 time_shift_bins=64, 
                 coord_x_bins=32, 
                 coord_y_bins=24,
                 max_slider_repeats=4, 
                 spinner_duration_bins=16,
                 beat_length_bins=64,
                 slider_velocity_bins=32,
                 supported_time_signatures=[3, 4, 5, 6, 7],
                 slider_max_relative_delta=128):

        self.time_shift_bins = time_shift_bins
        self.coord_x_bins = coord_x_bins
        self.coord_y_bins = coord_y_bins
        self.max_slider_repeats = max_slider_repeats
        self.spinner_duration_bins = spinner_duration_bins
        self.beat_length_bins = beat_length_bins
        self.slider_velocity_bins = slider_velocity_bins
        self.supported_time_signatures = sorted(list(set(supported_time_signatures))) # Ensure unique and sorted

        # Max relative delta for slider anchor coordinates
        self.slider_max_relative_delta = slider_max_relative_delta

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
        """Adds all defined tokens."""
        print("Building Vocabulary...")
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0

        # 1. Special Tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_id = self._add_token(self.pad_token)
        self.sos_id = self._add_token(self.sos_token)
        self.eos_id = self._add_token(self.eos_token)
        print(f"Added Special Tokens (PAD={self.pad_id}, SOS={self.sos_id}, EOS={self.eos_id})")


        # 2. Time Shift Tokens
        print(f"Adding {self.time_shift_bins} Time Shift Tokens...")
        for i in range(self.time_shift_bins):
            self._add_token(f"TIME_SHIFT_{i}")


        # 3. Coordinate Tokens (Used for absolute and relative)
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
            "START_SLIDER_C", # Catmull
            "ADD_SLIDER_ANCHOR", # For relative anchor points
            "PLACE_SPINNER",
        ]
        for token in action_tokens:
            self._add_token(token)

        # 6. Slider End Tokens (with repeats)
        print(f"Adding Slider End Tokens (0 to {self.max_slider_repeats} repeats)...")
        for i in range(self.max_slider_repeats + 1):
             self._add_token(f"END_SLIDER_{i}R")

        # 7. Spinner End Tokens (with duration bins)
        print(f"Adding Spinner End Tokens ({self.spinner_duration_bins} duration bins)...")
        for i in range(self.spinner_duration_bins):
             self._add_token(f"END_SPINNER_DUR_{i}")

        # --- 6. Timing Point Tokens
        print("Adding Timing Point Tokens...")
        self._add_token("UNINHERITED_TIMING_POINT") # Signals Beat Length change
        self._add_token("INHERITED_TIMING_POINT")   # Signals SV change

        print(f"Adding {self.beat_length_bins} Beat Length Tokens...")
        for i in range(self.beat_length_bins):
            self._add_token(f"BEAT_LENGTH_BIN_{i}")

        print(f"Adding {self.slider_velocity_bins} Slider Velocity Tokens...")
        for i in range(self.slider_velocity_bins):
            self._add_token(f"SLIDER_VELOCITY_BIN_{i}")

        print(f"Adding Time Signature Tokens ({len(self.supported_time_signatures)} supported)...")
        for ts in self.supported_time_signatures:
            self._add_token(f"TIME_SIGNATURE_{ts}")

        print(f"--- Vocabulary Built! Total size: {self.vocab_size} ---")


    def save(self, filepath="vocabulary.json"):
        """Saves the vocabulary mappings and parameters to a JSON file."""
        data = {
            'token_to_id': self.token_to_id,
            # Store parameters used to build the vocab
            'time_shift_bins': self.time_shift_bins,
            'coord_x_bins': self.coord_x_bins,
            'coord_y_bins': self.coord_y_bins,
            'max_slider_repeats': self.max_slider_repeats,
            'spinner_duration_bins': self.spinner_duration_bins,
            'slider_max_relative_delta': self.slider_max_relative_delta, # Keep
            'beat_length_bins' : self.beat_length_bins,
            'slider_velocity_bins' : self.slider_velocity_bins,
            'supported_time_signatures' : self.supported_time_signatures,
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

    def get_id(self, token_str):
        """Gets the ID for a token string."""
        return self.token_to_id.get(token_str) 

    def get_token(self, token_id):
        """Gets the token string for an ID."""
        return self.id_to_token.get(token_id) 

    def __len__(self):
        """Returns the total size of the vocabulary."""
        return self.vocab_size
    
    @classmethod
    def load(cls, filepath="vocabulary.json"):
        """Loads the vocabulary from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract parameters from saved data, providing defaults if missing
            time_shift_bins = data.get('time_shift_bins', 64)
            coord_x_bins = data.get('coord_x_bins', 32)
            coord_y_bins = data.get('coord_y_bins', 24)
            max_slider_repeats = data.get('max_slider_repeats', 4)
            spinner_duration_bins = data.get('spinner_duration_bins', 16)
            slider_max_relative_delta = data.get('slider_max_relative_delta', 128) 

            # Create a new instance using the loaded parameters
            instance = cls(time_shift_bins=time_shift_bins,
                            coord_x_bins=coord_x_bins,
                            coord_y_bins=coord_y_bins,
                            max_slider_repeats=max_slider_repeats,
                            spinner_duration_bins=spinner_duration_bins,
                            slider_max_relative_delta=slider_max_relative_delta,
                            beat_length_bins=data.get('beat_length_bins', 64),
                            slider_velocity_bins=data.get('slider_velocity_bins', 32),
                            supported_time_signatures=data.get('supported_time_signatures', [3, 4, 5, 6, 7]))
            
            # ... (keep optional verification logic) ...
            if instance.token_to_id != data['token_to_id']:
                 print("Warning: Loaded token_to_id map differs from the map rebuilt using loaded parameters. Using the rebuilt map.")

            # ... (keep restoring special tokens and other attributes) ...
            instance.pad_token = data.get('pad_token', '<PAD>')
            instance.sos_token = data.get('sos_token', '<SOS>')
            instance.eos_token = data.get('eos_token', '<EOS>')
            instance.pad_id = instance.get_id(instance.pad_token)
            instance.sos_id = instance.get_id(instance.sos_token)
            instance.eos_id = instance.get_id(instance.eos_token)
            instance.slider_max_relative_delta = slider_max_relative_delta

            print(f"Vocabulary loaded from {filepath}, size: {instance.vocab_size}")
            return instance
        # ... (keep except blocks for loading errors) ...
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
    # Create a new vocabulary (now excludes slider duration bins)
    vocab = BeatmapVocabulary() # Use defaults

    # Print some info
    print(f"\nVocabulary Size: {len(vocab)}")
    print(f"ID for '<PAD>': {vocab.pad_id}")
    print(f"ID for 'PLACE_CIRCLE': {vocab.get_id('PLACE_CIRCLE')}")
    print(f"ID for 'COORD_X_15': {vocab.get_id('COORD_X_15')}")
    print(f"ID for 'ADD_SLIDER_ANCHOR': {vocab.get_id('ADD_SLIDER_ANCHOR')}")
    print(f"Token for ID 50: {vocab.get_token(50)}") # Example ID, might change

    # Save the vocabulary
    vocab.save("my_beatmap_vocab_v4.json")
    # Load the vocabulary
    loaded_vocab = BeatmapVocabulary.load("my_beatmap_vocab_v4.json")
    if loaded_vocab:
        print(f"\nLoaded Vocabulary Size: {len(loaded_vocab)}")