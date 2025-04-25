from .models import (Beatmap, Vector2, HitCircleOsu, SliderOsu,
    SpinnerOsu, ControlPoint, HitComboOsu)
from .enums import FileSection, HitObjectType
from decimal import Decimal
from numpy import float32

class BeatmapOsu(Beatmap):

    def __init__(self, filename=None):
        super().__init__()
        if filename:
            self.filename = filename
            lines = None
            with open(filename, "r", encoding="utf-8-sig") as f:
                lines = f.read().splitlines()
            if lines:
                self._process_headers(lines)
                self._parse(lines)
                self._sort_objects()

    def _process_headers(self, lines):
        current_section = FileSection.UNKNOWN
        ar_is_od = True
        i = 0
        try:
            try:
                line = lines[i]
                if line.index("osu file format") == 0:
                    self.beatmap_version = int(line[line.rindex("v") + 1:])
            except ValueError as e:
                print(f"Missing file format for {self.filename}")
            while (i := i + 1) < len(lines):
                line = lines[i].strip()
                if len(line) == 0 or line.startswith("//"):
                    continue
                left = ""
                right = ""
                if current_section != FileSection.HITOBJECTS:
                    kv = line.split(":", 1)
                    if len(kv) > 1:
                        left = kv[0].strip()
                        right = kv[1].strip()
                    elif line[0] == "[":
                        try:
                            current_section = FileSection[
                                                line.strip("[]").upper()
                                              ]
                        except:
                            pass
                        continue
                if current_section == FileSection.TIMINGPOINTS:
                    try:
                        split = line.split(",")
                        if len(split) < 2:
                            continue
                        offset = float(split[0].strip())
                        beat_length = float(split[1].strip())
                        timing_change = True
                        if len(split) > 6:
                            timing_change = split[6][0] == "1"
                        tp = ControlPoint(offset, beat_length, timing_change)
                        self.control_points.append(tp)
                    except Exception as e:
                        print(f"Error parsing timing points for"
                              f"{self.filename}\n{e}")
                        pass
                elif current_section == FileSection.METADATA:
                    if left == "Artist":
                        self.artist = right
                    elif left == "ArtistUnicode":
                        self.artist_unicode = right
                    elif left == "Title":
                        self.title = right
                    elif left == "TitleUnicode":
                        self.title_unicode = right
                    elif left == "Creator":
                        self.creator = right
                    elif left == "Version":
                        self.version = right
                    elif left == "Tags":
                        self.tags = right
                    elif left == "Source":
                        self.source = right
                    elif left == "BeatmapID":
                        if self.beatmap_id == 0:
                            self.beatmap_id = int(right)
                    elif left == "BeatmapSetID":
                        if self.beatmapset_id == -1:
                            self.beatmapset_id = int(right)
                elif current_section == FileSection.DIFFICULTY:
                    if left == "HPDrainRate":
                        if self.beatmap_version >= 13:
                            self.hp = min(float32(10), max(float32(0),
                                                           float32(right)))
                        else:
                            self.hp = float32(min(10, max(0, int(right))))
                    elif left == "CircleSize":
                        if self.beatmap_version >= 13:
                            self.cs = min(float32(10), max(float32(0),
                                                           float32(right)))
                        else:
                            self.cs = float32(min(10, max(0, int(right))))
                    elif left == "OverallDifficulty":
                        if self.beatmap_version >= 13:
                            self.od = min(float32(10), max(float32(0),
                                                           float32(right)))
                        else:
                            self.od = float32(min(10, max(0, int(right))))
                        if ar_is_od:
                            self.ar = self.od
                    elif left == "SliderMultiplier":
                        self.slider_multiplier = max(0.4, min(3.6,
                                                              float(right)))
                    elif left == "SliderTickRate":
                        self.slider_tick_rate = max(0.5, min(8, float(right)))
                    elif left == "ApproachRate":
                        if self.beatmap_version >= 13:
                            self.ar = min(float32(10), max(float32(0),
                                                           float32(right)))
                        else:
                            self.ar = float32(min(10, max(0, int(right))))
                        ar_is_od = False
        except Exception as e:
            print(f"An error occured while processing {self.filename}\n{e}")
        self.slider_scoring_point_distance = (100 * self.slider_multiplier /
                                              self.slider_tick_rate)

    def _sort_objects(self):
        self.hit_objects.sort(key=lambda ho: ho.end_time)

    def beat_length_at(self, time):
        if len(self.control_points) == 0:
            return 0
        point = 0
        sample_point = 0
        for i in range(len(self.control_points)):
            if self.control_points[i].offset <= time:
                if self.control_points[i].timing_change:
                    point = i
                else:
                    sample_point = i
        mult = 1.0
        if (sample_point > point and
            self.control_points[sample_point].beat_length < 0):
            mult = self.control_points[sample_point].bpm_multiplier()
        return self.control_points[point].beat_length * mult

    def _parse(self, lines):
        current_section = FileSection.UNKNOWN
        for line in lines:
            if (len(line) == 0 or line.startswith(" ") or line.startswith("_")
                or line.startswith("//")):
                continue
            if line[0] == "[":
                try:
                    current_section = FileSection[line.strip("[]").upper()]
                except:
                    pass
                continue
            kv = line.split(":", 1)
            if len(kv) > 1:
                key = kv[0].strip()
                val = kv[1].strip()
            if current_section == FileSection.HITOBJECTS:
                split = line.split(",")
                obj_type = (HitObjectType(int(split[3])) &
                            ~HitObjectType.COLOURHAX)
                x = int(max(0, min(512, Decimal(split[0]))))
                y = int(max(0, min(512, Decimal(split[1]))))
                pos = Vector2(x, y)
                time = int(Decimal(split[2]))

                """
                if any(combo_type in obj_type for combo_type in (HitObjectType.NEWCOMBO, HitObjectType.NORMALNEWCOMBO, HitObjectType.SLIDERNEWCOMBO)):
                    h = HitComboOsu(time, time)
                """
                if HitObjectType.NORMAL in obj_type:
                    if HitObjectType.NEWCOMBO in obj_type:
                        h = HitCircleOsu(True, pos, time, time)
                    else:
                        h = HitCircleOsu(False, pos, time, time)
                    self.circle_count += 1
                elif HitObjectType.SLIDER in obj_type:
                    length = 0
                    curve_data = split[5].split("|", 1)
                    curve_type = curve_data[0]  
                    curve_points = [
                        Vector2(*map(int, point.split(":")))
                        for point in curve_data[1].split("|")
                    ]
                    
                    repeat_count = int(split[6])
                    if len(split) > 7:
                        length = float(split[7])
                    beat_length = self.beat_length_at(time)
                    end_time = (time + int(length /
                        (100 * self.slider_multiplier) * beat_length) *
                        repeat_count)
                    if HitObjectType.NEWCOMBO in obj_type:
                        h = SliderOsu(True, pos, time, end_time, max(1, repeat_count), length, curve_type, curve_points)
                    else:
                        h = SliderOsu(False, pos, time, end_time, max(1, repeat_count), length, curve_type, curve_points)
                    self.slider_count += 1
                elif HitObjectType.SPINNER in obj_type:
                    end_time = int(split[5])
                    if HitObjectType.NEWCOMBO in obj_type:
                        h = SpinnerOsu(True, time, end_time)
                    else:
                        h = SpinnerOsu(False, time, end_time)
                    self.spinner_count += 1
                if h:
                    self.hit_objects.append(h)


if __name__ == '__main__':
    beatmap = BeatmapOsu("./harmony/harmony-hard.osu")
    print("Artist: ", beatmap.artist)
    print("Title: ", beatmap.title)
    print("Creator: ", beatmap.creator)
    print("Version: ", beatmap.version)
    print("Beatmap ID: ", beatmap.beatmap_id)
    print("Beatmap Set ID: ", beatmap.beatmapset_id)
    print("HP: ", beatmap.hp)
    print("CS: ", beatmap.cs)
    print("OD: ", beatmap.od)
    print("AR: ", beatmap.ar)
    print("Slider Multiplier: ", beatmap.slider_multiplier)
    print("Slider Tick Rate: ", beatmap.slider_tick_rate)
    print("Slider Scoring Point Distance: ", beatmap.slider_scoring_point_distance)
    print("Control Points: ", beatmap.control_points)
    print("Hit Objects: ", beatmap.hit_objects)
    print("Circle Count: ", beatmap.circle_count)
    print("Slider Count: ", beatmap.slider_count)
    print("Spinner Count: ", beatmap.spinner_count)
    print("Hit Objects Count: ", len(beatmap.hit_objects))
