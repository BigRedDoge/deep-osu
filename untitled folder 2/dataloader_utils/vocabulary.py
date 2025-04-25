import math
import json

class BeatmapVocabulary:
    def __init__(self, time_shift_bins=64, coord_x_bins=32, coord_y_bins=24,
                 max_slider_repeats=4, spinner_duration_bins=16, slider_duration_bins=32): # Default to 32 bins for sliders

        self.time_shift_bins = time_shift_bins
        self.coord_x_bins = coord_x_bins
        self.coord_y_bins = coord_y_bins
        self.max_slider_repeats = max_slider_repeats
        self.spinner_duration_bins = spinner_duration_bins
        self.slider_duration_bins = slider_duration_bins

        # Max relative delta for slider anchor coordinates
        self.slider_max_relative_delta = 128

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
        # Warn if token string already exists but points to a different ID (should not happen with this logic)
        elif self.token_to_id[token_str] != self.vocab_size -1 and self.id_to_token.get(self.token_to_id[token_str]) == token_str:
             # This case means the token was already added correctly.
             pass
        elif self.token_to_id[token_str] != self.vocab_size -1 :
             print(f"Warning: Token '{token_str}' already exists with ID {self.token_to_id[token_str]}, but current vocab size is {self.vocab_size}. Check for duplicates.")

        return self.token_to_id[token_str]

    def _build_vocabulary(self):
        """Systematically adds all defined tokens."""
        print("Building Vocabulary...")
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0

        # 1. Special Tokens 
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

        # 5. Slider Duration Tokens (NEW)
        print(f"Adding {self.slider_duration_bins} Slider Duration Tokens...")
        for i in range(self.slider_duration_bins):
             self._add_token(f"SLIDER_DUR_{i}")

        # 6. Slider End Tokens (with repeats)
        print(f"Adding Slider End Tokens (0 to {self.max_slider_repeats} repeats)...")
        for i in range(self.max_slider_repeats + 1):
             self._add_token(f"END_SLIDER_{i}R")

        # 7. Spinner End Tokens (with duration bins)
        print(f"Adding Spinner End Tokens ({self.spinner_duration_bins} duration bins)...")
        for i in range(self.spinner_duration_bins):
             self._add_token(f"END_SPINNER_DUR{i}")

        print(f"--- Vocabulary Built! Total size: {self.vocab_size} ---")

    def get_id(self, token_str):
        """Gets the ID for a token string."""
        return self.token_to_id.get(token_str) # Returns None if not found

    def get_token(self, token_id):
        """Gets the token string for an ID."""
        return self.id_to_token.get(token_id) # Returns None if not found

    def __len__(self):
        """Returns the total size of the vocabulary."""
        return self.vocab_size

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
            'slider_duration_bins': self.slider_duration_bins, # Save new param
            'slider_max_relative_delta': self.slider_max_relative_delta,
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

            # Extract parameters from saved data, providing defaults if missing
            time_shift_bins = data.get('time_shift_bins', 64)
            coord_x_bins = data.get('coord_x_bins', 32)
            coord_y_bins = data.get('coord_y_bins', 24)
            max_slider_repeats = data.get('max_slider_repeats', 4)
            spinner_duration_bins = data.get('spinner_duration_bins', 16)
            slider_duration_bins = data.get('slider_duration_bins', 32) # Load new param
            slider_max_relative_delta = data.get('slider_max_relative_delta', 128)

            # Create a new instance using the loaded parameters
            # This automatically calls _build_vocabulary with the correct params
            instance = cls(time_shift_bins=time_shift_bins,
                           coord_x_bins=coord_x_bins,
                           coord_y_bins=coord_y_bins,
                           max_slider_repeats=max_slider_repeats,
                           spinner_duration_bins=spinner_duration_bins,
                           slider_duration_bins=slider_duration_bins)

            # Optionally, verify the loaded token_to_id matches the rebuilt one
            # This is useful for ensuring consistency after code changes
            if instance.token_to_id != data['token_to_id']:
                 print("Warning: Loaded token_to_id map differs from the map rebuilt using loaded parameters. Using the rebuilt map.")
                 # Or, force using the loaded map if that's desired:
                 # instance.token_to_id = data['token_to_id']
                 # instance.id_to_token = {v: k for k, v in instance.token_to_id.items()}
                 # instance.vocab_size = len(instance.token_to_id)

            # Restore special tokens from saved data if needed (though _build sets defaults)
            instance.pad_token = data.get('pad_token', '<PAD>')
            instance.sos_token = data.get('sos_token', '<SOS>')
            instance.eos_token = data.get('eos_token', '<EOS>')
            instance.pad_id = instance.get_id(instance.pad_token)
            instance.sos_id = instance.get_id(instance.sos_token)
            instance.eos_id = instance.get_id(instance.eos_token)

            # Restore other attributes if necessary (already done via __init__)
            instance.slider_max_relative_delta = slider_max_relative_delta

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


# Example usage remains the same for demonstration
if __name__ == "__main__":
    # Create a new vocabulary (now includes slider duration bins)
    vocab = BeatmapVocabulary(slider_duration_bins=32)

    # Print some info
    print(f"\nVocabulary Size: {len(vocab)}")
    print(f"ID for '<PAD>': {vocab.pad_id}")
    print(f"ID for 'PLACE_CIRCLE': {vocab.get_id('PLACE_CIRCLE')}")
    print(f"ID for 'COORD_X_15': {vocab.get_id('COORD_X_15')}")
    print(f"ID for 'SLIDER_DUR_10': {vocab.get_id('SLIDER_DUR_10')}") # Test new token
    print(f"Token for ID 50: {vocab.get_token(50)}") # Example

    # Save the vocabulary
    vocab.save("my_beatmap_vocab_v3.json")

    # Load the vocabulary (demonstration)
    loaded_vocab = BeatmapVocabulary.load("my_beatmap_vocab_v3.json")
    if loaded_vocab:
        print(f"\nLoaded Vocabulary Size: {len(loaded_vocab)}")
        print(f"Loaded ID for 'PLACE_CIRCLE': {loaded_vocab.get_id('PLACE_CIRCLE')}")
        print(f"Loaded ID for 'SLIDER_DUR_10': {loaded_vocab.get_id('SLIDER_DUR_10')}")
        print(f"Token for ID {loaded_vocab.pad_id}: {loaded_vocab.get_token(loaded_vocab.pad_id)}")
        print(f"Loaded slider duration bins: {loaded_vocab.slider_duration_bins}")

