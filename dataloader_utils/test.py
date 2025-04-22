from osu_to_token import OsuToToken
from vocabulary import BeatmapVocabulary
from quantizers import Quantizers

vocab = BeatmapVocabulary(
    time_shift_bins=64, #128
    coord_x_bins=64, #64
    coord_y_bins=48, #48
    max_slider_repeats=4,
    spinner_duration_bins=16
)

quantizers = Quantizers(vocab)


osu_tokens = OsuToToken(
    osu_file_path="beatmapparser/harmony/harmony-hard.osu",
    quantizers=quantizers, 
    vocab=vocab,
    max_seq_len=4096
)

#print("Token IDs:", osu_tokens.token_id_sequence)
for i, token_id in enumerate(osu_tokens.token_id_sequence):
    token_str = vocab.get_token(token_id)
    print(f"Token ID {i}: {token_id} -> Token String: {token_str}")
print("Token IDs Length:", len(osu_tokens.token_id_sequence))
print("Vocabulary Size:", vocab.vocab_size)
