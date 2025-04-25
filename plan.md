Autoregressive Beatmap Generation Plan (v2)
===========================================

This document outlines the plan to create an autoregressive model for generating .osu beatmaps, using a CNN Encoder and Transformer Decoder architecture. This version incorporates the use of the osu-parsers Node.js library for parsing input .osu files and encoding generated data back into the .osu format, including support for timing point tokenization.

Phase 1: Data Representation - Defining the Language
----------------------------------------------------

Goal: Create the vocabulary and grammar the model will use to read and write beatmaps. Choices here directly impact the model's learning ability and output quality.

### Step 1.1: Define the Core Information to Capture

-   Explanation: To faithfully represent beatmap gameplay and structure, we must capture all essential elements.

-   Map Boundaries (SOS, EOS):

-   Rationale: Standard practice in sequence modeling. SOS (Start Of Sequence) provides a starting token for the decoder during generation. EOS (End Of Sequence) allows the model to learn when a map is complete, enabling variable-length generation.

-   Time Progression (TIME_SHIFT):

-   Rationale: Represents the temporal relationship between events. Encoding the difference in time makes the representation relative and potentially easier for the model to learn rhythmic patterns compared to absolute timestamps. It captures the "wait time" before the next action. Includes shifts between both hit objects and timing points.

-   Object Types (PLACE_CIRCLE, START_SLIDER_*, PLACE_SPINNER):

-   Rationale: These tokens represent the fundamental actions or object types available in osu!standard. Separating them allows the model to learn the distinct characteristics and subsequent parameters associated with each type.

-   Object Positions (COORD_X, COORD_Y):

-   Rationale: Essential for defining where interactions occur. Using separate X and Y tokens keeps the vocabulary smaller than combining them. Normalizing coordinates (0-1) makes the representation independent of playfield resolution. Using relative coordinates for slider anchors is hypothesized to help learn shape invariance (see Step 1.2).

-   Slider Paths (START_SLIDER(type), ADD_SLIDER_ANCHOR, END_SLIDER(repeats)):

-   Rationale: Sliders are complex. We need tokens to signal the start (including type like Linear, Bezier), each anchor point defining the curve, and the end (including repeats). This structured approach breaks down slider creation into manageable steps for the sequence model.

-   Spinner Durations (END_SPINNER(duration)):

-   Rationale: Spinners primarily require a duration. An end token carrying quantized duration information is efficient. Position is fixed (center).

-   Combo Management (NEW_COMBO):

-   Rationale: New combos affect scoring and visual grouping. A simple flag token before an object is placed is a straightforward way to represent this optional attribute.

-   Timing Points (UNINHERITED_*, INHERITED_*, BEAT_LENGTH_BIN_*, SLIDER_VELOCITY_BIN_*, TIME_SIGNATURE_*, EFFECT_KIAI_*):

-   Rationale: To capture rhythmic and difficulty variations within a map, changes in BPM (via beat length), Slider Velocity (SV), Time Signature, and Kiai need to be encoded. Separate tokens indicate the type of timing point and subsequent tokens carry the quantized value or state.

### Step 1.2: Design the Token Vocabulary

-   Rationale: We aim for a balance between expressiveness and vocabulary size. A smaller vocabulary is generally easier for models to learn, but must still capture all necessary information.

-   Quantization:

-   Rationale: Converts continuous values (time, position, beat length, SV) into discrete bins. This transforms the prediction task into classification (predicting the right bin/token), which is often easier for sequence models than regressing continuous values directly. It's essential for managing vocabulary size. The choice of quantization scale (linear vs. logarithmic) depends on the perceptual importance of the value (e.g., log for time/beat length where relative changes matter more, linear for position/SV where absolute changes might be more relevant).

-   Separation (Action/Parameter Tokens):

-   Rationale: Using tokens like PLACE_CIRCLE followed by COORD_X(15) and COORD_Y(22) instead of a single combined token CIRCLE_X15_Y22 drastically reduces vocabulary size (e.g., 1 + 32 + 32 tokens vs. 1 * 32 * 32 tokens). It encourages the model to learn the compositional structure (action -> parameters). The trade-off is potentially longer sequences.

-   Vocabulary Categories (Detailed):

-   Special Tokens: Standard practice for sequence modeling (Padding, Start, End).

-   Time Tokens (TIME_SHIFT(bin)): Logarithmic quantization captures the higher sensitivity to small timing changes at short intervals (e.g., 1/4 vs 1/3 beat) compared to the same absolute difference at longer intervals.

-   Coordinate Tokens (COORD_X(bin), COORD_Y(bin)): Linear quantization is generally sufficient for screen position. Using the same bins for absolute start positions and relative anchor deltas (after appropriate normalization/scaling) keeps the vocabulary smaller, assuming the model can learn the context from the preceding tokens (START_SLIDER_* vs ADD_SLIDER_ANCHOR). 32x24 bins offer a reasonable precision trade-off (~16px on X, ~16px on Y).

-   Hit Object Action/Type Tokens: Clear, distinct tokens for each core action or object type. Slider types (L/B/P/C) are included for shape information.

-   Timing Point Tokens:

-   UNINHERITED/INHERITED: Distinguishes points that reset timing (BPM) from those that only modify SV relative to the current BPM. Essential for correct timing reconstruction.

-   BEAT_LENGTH_BIN: Logarithmic quantization reflects that BPM changes are often perceived multiplicatively (e.g., 120->180 BPM is a bigger jump than 300->360 BPM in terms of feel).

-   SLIDER_VELOCITY_BIN: Linear quantization is likely sufficient as SV changes are often additive or simple multipliers in practice.

-   TIME_SIGNATURE: Only include common signatures to avoid excessive rare tokens. The model might struggle with very unusual meters anyway.

-   EFFECT_KIAI: Simple ON/OFF state tokens are sufficient.

-   Vocabulary Size Estimate: A concrete estimate helps gauge model complexity. ~350 tokens is manageable for modern Transformer architectures.

### Step 1.3: Implement Tokenizer (.osu -> Token Sequence)

-   Tooling (Node.js osu-parsers):

-   Rationale: osu-parsers is a robust, well-maintained library specifically designed for accurate parsing of .osu files, including complex slider paths and timing point details. Using it via a Node.js subprocess ensures high fidelity input data compared to potentially less complete Python parsers. Communication via JSON provides a structured way to pass complex data between languages.

-   Input/Output: Standard file input, list/tensor output suitable for deep learning frameworks.

-   Process:

1.  Execute Node Parser: Necessary step to leverage the chosen library.

2.  Receive JSON: Standard inter-process communication method.

3.  Combine & Sort Events:

-   Rationale: Treat both hit objects and timing points as events in a single timeline. Sorting chronologically ensures the TIME_SHIFT token correctly represents the time since the immediately preceding event, regardless of its type. Processing control points before hit objects at the same timestamp matches how osu! applies timing changes.

1.  Initialize: Standard sequence start.

2.  Iterate Sorted Events:

-   Time Shift: Calculated first for every event to establish its temporal position relative to the previous one.

-   Control Point Event Logic: Emit tokens corresponding to the changes introduced by this control point (new BPM, new SV, Kiai toggle). This requires quantizing the relevant values (beat length, SV).

-   Hit Object Event Logic: Emit tokens describing the object placement and properties. For sliders, the sequence (START, coords, ADD_ANCHOR, relative coords..., END) explicitly defines the structure. Calculating relative anchor coordinates requires tracking the previous absolute position.

-   Update Time: Essential for calculating the next TIME_SHIFT.

1.  Finalize: Add EOS, pad to fixed length for batch processing in the model.

2.  Return: The final token sequence.

### Step 1.4: Implement Detokenizer (Token Sequence -> .osu)

-   Tooling (Python Reconstruction + Node.js Encoding):

-   Rationale: Reconstructing the logical structure (lists of hit objects and control points) is easier in Python where the main model logic resides. However, correctly formatting the final .osu string, especially the [TimingPoints] and [HitObjects] sections with their specific syntax and interdependencies (like slider duration depending on timing), is complex and error-prone. Leveraging osu-parsers' BeatmapEncoder via a Node.js script ensures the output is valid and correctly formatted according to osu! standards. This separates the logical reconstruction (Python) from the strict formatting rules (Node.js).

-   Input/Output: Token sequence in, .osu string out.

-   Process:

1.  Stage 1: Reconstruct Data (Python):

-   Iterate through the cleaned token sequence.

-   Maintain state (current time, pending objects, current timing values).

-   Recognize tokens and dequantize values using the QuantizationManager.

-   Build up hit_objects and control_points lists containing dictionaries of reconstructed properties. Handling pending states (like accumulating slider points before END_SLIDER) is key. Finalizing pending timing points before time shifts or new object starts ensures correct association.

1.  Stage 2: Format .osu (Node.js):

-   Pass the reconstructed lists as JSON to the Node script.

-   Node script uses osu-parsers to create a Beatmap object.

-   Populates the Beatmap object with metadata, difficulty, control points (creating TimingPoint, DifficultyPoint, etc. objects), and hit objects (creating HittableObject, SlidableObject, etc., and reconstructing SliderPath).

-   Crucially calls applyDefaults(): This step within osu-parsers calculates derived properties like slider durations and tick points based on the added control points and difficulty settings.

-   Uses BeatmapEncoder to generate the final, correctly formatted .osu string. Python returns this string.

Phase 2: Data Loading and Preprocessing
---------------------------------------

Goal: Prepare the raw audio and parsed beatmap data for model training.

### Step 2.1: Update Dataset and DataLoader

-   BeatmapDataset (torch.utils.data.Dataset):

-   Rationale: Standard PyTorch class for handling data loading. Each item represents one beatmap-audio pair.

-   __getitem__(idx): Defines the loading and preprocessing pipeline for a single sample.

1.  Audio Features: Mel Spectrogram is a common and effective audio representation for deep learning, capturing frequency content over time.

2.  Parse & Tokenize: Uses the OsuToToken instance (from Phase 1) to convert the .osu file into the target token sequence the model needs to predict.

3.  Difficulty Features: AR, CS, OD, HP directly influence map playability and structure. Providing them as input conditions the model's generation. Requires the Node parser to extract them.

4.  Pad/Truncate: Necessary to create batches of uniform tensor shapes for efficient GPU processing. Padding masks inform the model which parts of the sequence are real data versus padding.

-   DataLoader (torch.utils.data.DataLoader):

-   Rationale: Standard PyTorch utility. Handles creating mini-batches from the Dataset, shuffling data for better training generalization, and potentially parallelizing data loading.

Phase 3: Model Architecture (CNN Encoder + Transformer Decoder)
---------------------------------------------------------------

Goal: Define the neural network that learns the mapping from audio and difficulty to the beatmap token sequence.

-   Step 3.1: CNN Encoder:

-   Rationale: Processes the audio spectrogram. CNNs excel at capturing local patterns (like onsets, rhythmic motifs). Using strides reduces the sequence length of the audio representation, making the subsequent Transformer computationally less expensive (Transformers have quadratic complexity w.r.t. sequence length). Outputs a compressed, contextualized representation (audio_memory).

-   Step 3.2: Difficulty Embedding:

-   Rationale: Neural networks work with high-dimensional vectors. A simple linear layer transforms the low-dimensional difficulty features (e.g., 4 numbers) into the model's internal dimension (d_model), allowing them to be combined with other features.

-   Step 3.3: Token Embedding:

-   Rationale: Converts discrete token IDs into continuous vector representations that the network can process. nn.Embedding is the standard layer for this. padding_idx ensures PAD tokens don't affect learning.

-   Step 3.4: Positional Encoding:

-   Rationale: Transformers themselves don't inherently understand sequence order. Positional encoding injects information about the position of each token (absolute or relative) into its embedding, allowing the model to consider sequence structure. Sinusoidal encoding is a common, fixed method; learned embeddings are an alternative.

-   Step 3.5: Transformer Decoder:

-   Rationale: The core sequence generation component. It attends to the audio context (audio_memory via cross-attention) and the previously generated tokens (tgt via masked self-attention) to predict the next token in the sequence. The masked self-attention prevents the model from "cheating" by looking at future tokens during training. Multiple layers allow learning complex dependencies. Padding masks prevent attention calculations on padding tokens.

-   Step 3.6: Output Head:

-   Rationale: Maps the Transformer's final hidden state (in d_model dimensions) back to the size of the vocabulary. The output represents the model's predicted probability distribution (logits) over all possible next tokens.

-   Step 3.7: Combining Difficulty:

-   Rationale: Adding the difficulty embedding to the token embeddings injects the difficulty context into every step of the decoding process. This allows the model to generate different patterns or densities based on the requested difficulty settings (AR, CS, OD, HP). Broadcasting (unsqueeze + expand) is needed to match dimensions.

Phase 4: Training
-----------------

Goal: Optimize the model's parameters using the prepared data.

-   Step 4.1: Loss Function (nn.CrossEntropyLoss):

-   Rationale: Standard loss function for multi-class classification problems (predicting the next token from the vocabulary). It combines LogSoftmax and Negative Log Likelihood, suitable for comparing the model's output logits with the target token IDs. ignore_index=PAD_ID is crucial to prevent calculating loss on padding tokens.

-   Step 4.2: Optimizer (torch.optim.AdamW):

-   Rationale: AdamW is an effective and widely used optimizer for training deep neural networks, particularly Transformers. It adapts learning rates per parameter and includes decoupled weight decay, often leading to better generalization than standard Adam.

-   Step 4.3: Learning Rate Scheduler:

-   Rationale: Adjusting the learning rate during training is critical. Starting high helps escape local minima, while decreasing it later allows for finer convergence. Schedulers automate this process (e.g., reducing LR when validation loss plateaus, or using a predefined warmup/decay schedule).

-   Step 4.4: Training Loop:

-   Rationale: Implements the standard training procedure.

-   Teacher Forcing: During training, the decoder receives the ground truth previous token as input (shifted right), not its own prediction. This stabilizes training early on.

-   Masks: Essential for the Transformer decoder (causal mask for self-attention, padding masks for self- and cross-attention) to ensure correct information flow and prevent attending to padding.

-   Forward/Backward Pass: Standard steps for calculating output, loss, and gradients.

-   Gradient Clipping: Prevents exploding gradients, a common issue in training deep networks, by capping the norm of the gradients.

-   Optimizer/Scheduler Steps: Update model weights and learning rate.

Phase 5: Inference (Generation)
-------------------------------

Goal: Use the trained model to create new beatmaps.

-   Step 5.1: Setup:

-   Rationale: model.eval() disables dropout/batchnorm updates. torch.no_grad() disables gradient calculation, saving memory and computation as gradients aren't needed for inference. Pre-computing audio encoding is efficient as it only needs to be done once per generation.

-   Step 5.2: Autoregressive Generation Loop:

-   Rationale: Generation happens one token at a time. The model predicts the next token based on the audio, difficulty, and all previously generated tokens. This is the "autoregressive" property.

-   Initialize with SOS_ID.

-   In the loop: Feed the current sequence to the decoder, get logits for the next token.

-   Sampling Strategy: Choose how to select the next token from the probability distribution (logits). Greedy is simple but repetitive. Top-k/Top-p (Nucleus) offer a balance between coherence and diversity by sampling from a truncated distribution. Beam search explores multiple possibilities but is more complex. Top-p is often a good default.

-   Append the sampled token and repeat until EOS_ID or max_length.

-   Step 5.3: Detokenization:

-   Rationale: Convert the generated sequence of token IDs back into a human-readable and usable format.

-   Clean the sequence (remove special tokens).

-   Use the TokenToOsu instance (from Phase 1) and its get_osu_string method. This handles the two-stage process: Python reconstructs the structured data (hit objects, control points), and the Node.js script formats it into the final .osu string using osu-parsers. This ensures the output is valid.

This updated plan provides a comprehensive approach, incorporating the osu-parsers library and handling timing point information throughout the tokenization and detokenization process.