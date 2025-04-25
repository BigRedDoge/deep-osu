import { BeatmapDecoder } from 'osu-parsers'

const path = '/Users/sean/Documents/deep-osu/dataloader_utils/beatmapparser/harmony/harmony-hard.osu';

console.log("test");
// This is optional and true by default.
const shouldParseSb = false;

async function parseBeatmap() {
    const decoder = new BeatmapDecoder();
    const beatmap1 = await decoder.decodeFromPath(path, shouldParseSb);
    console.log("test");
    console.log(beatmap1);
}

parseBeatmap();