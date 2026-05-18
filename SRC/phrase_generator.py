from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "MUESTRAS"
OUTPUT_AUDIO_DIR = PROJECT_ROOT / "GENERADAS" / "TRAIN" / "AUDIO"
OUTPUT_LABELS_DIR = PROJECT_ROOT / "GENERADAS" / "TRAIN" / "LABELS"

TARGET_SR = 48_000
PEAK_TARGET = 0.98
DEFAULT_BPM = 120
DEFAULT_PHRASE_COUNT = 100
GRID_STEPS = 16
BEATS_PER_PHRASE = 4
MIN_EVENTS_PER_PHRASE = 4
MAX_EVENTS_PER_PHRASE = 12
FADE_SECONDS = 0.005
TAIL_PADDING_SECONDS = 1.25
MICROTIMING_SECONDS_RANGE = (-0.005, 0.005)
INSTRUMENT_CLASSES = ("KD", "SD", "T12", "T14", "T16")
FILL_PROBABILITY = 0.72
FINAL_TWO_BEAT_FILL_PROBABILITY = 0.28
GHOST_NOTE_GAIN_RANGE = (0.22, 0.42)
ACCENT_GAIN_RANGE = (0.82, 1.0)
TOM_FILL_GAIN_RANGE = (0.58, 0.88)
FILL_TOM_PATH = ("T12", "T14", "T16")
FILL_MOTIFS = (
    (0, 2),
    (0, 1, 3),
    (0, 2, 3),
    (0, 1, 2),
)
FILL_RESOLUTIONS = ("KD", "KD", "SD", "KD+SD")


@dataclass(frozen=True)
class HitEvent:
    instrument: str
    sample_class: str
    beat: float
    gain: float = 1.0
    timing_offset_seconds: float = 0.0


@dataclass(frozen=True)
class GroovePattern:
    name: str
    kick_steps: tuple[int, ...]
    snare_steps: tuple[int, ...]
    optional_kick_steps: tuple[int, ...] = ()
    ghost_snare_steps: tuple[int, ...] = ()


GROOVE_PATTERNS = (
    GroovePattern(
        name="backbeat_basic",
        kick_steps=(0, 8),
        snare_steps=(4, 12),
        optional_kick_steps=(10, 14),
        ghost_snare_steps=(3, 7, 11, 15),
    ),
    GroovePattern(
        name="syncopated_kick",
        kick_steps=(0, 6, 8),
        snare_steps=(4, 12),
        optional_kick_steps=(10, 14),
        ghost_snare_steps=(2, 7, 11, 15),
    ),
    GroovePattern(
        name="four_on_floor_backbeat",
        kick_steps=(0, 4, 8, 12),
        snare_steps=(4, 12),
        optional_kick_steps=(14,),
        ghost_snare_steps=(7, 11, 15),
    ),
    GroovePattern(
        name="sparse_rock",
        kick_steps=(0, 10),
        snare_steps=(4, 12),
        optional_kick_steps=(6, 8, 14),
        ghost_snare_steps=(3, 11, 15),
    ),
)


@dataclass(frozen=True)
class RenderedPhrase:
    audio: np.ndarray
    labels: list[dict]
    sample_rate: int
    bpm: float
    tail_padding_seconds: float


def seconds_per_beat(bpm: float) -> float:
    if bpm <= 0:
        raise ValueError("BPM must be greater than zero.")
    return 60.0 / bpm


def list_wav_samples(sample_class: str, samples_dir: Path = SAMPLES_DIR) -> list[Path]:
    class_dir = samples_dir / sample_class
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Sample class directory not found: {class_dir}")

    wavs = sorted(class_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV samples found in: {class_dir}")
    return wavs


def choose_sample(
    sample_class: str,
    samples_dir: Path = SAMPLES_DIR,
    rng: random.Random | None = None,
) -> Path:
    samples = list_wav_samples(sample_class, samples_dir)
    if rng is None:
        return samples[0]
    return rng.choice(samples)


def load_hit(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, _ = librosa.load(path, sr=target_sr, mono=True, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    return audio.astype(np.float32, copy=False)


def apply_fades(audio: np.ndarray, sample_rate: int, fade_seconds: float = FADE_SECONDS) -> np.ndarray:
    if audio.size == 0:
        return audio

    fade_len = min(int(round(fade_seconds * sample_rate)), audio.size // 2)
    if fade_len <= 1:
        return audio

    faded = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    faded[:fade_len] *= fade_in
    faded[-fade_len:] *= fade_out
    return faded


def peak_normalize(audio: np.ndarray, target_peak: float = PEAK_TARGET) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32, copy=False)
    return (audio * (target_peak / peak)).astype(np.float32)


def render_phrase(
    events: Iterable[HitEvent],
    bpm: float = DEFAULT_BPM,
    samples_dir: Path = SAMPLES_DIR,
    sample_rate: int = TARGET_SR,
    seed: int = 7,
    tail_padding_seconds: float = TAIL_PADDING_SECONDS,
) -> RenderedPhrase:
    ordered_events = sorted(events, key=lambda event: event.beat)
    if not ordered_events:
        raise ValueError("At least one hit event is required.")

    beat_seconds = seconds_per_beat(bpm)
    rng = random.Random(seed)

    prepared_hits: list[tuple[HitEvent, Path, int, np.ndarray]] = []
    for event in ordered_events:
        sample_path = choose_sample(event.sample_class, samples_dir, rng)
        onset_seconds = max(0.0, event.beat * beat_seconds + event.timing_offset_seconds)
        onset_sample = int(round(onset_seconds * sample_rate))
        hit_audio = load_hit(sample_path, sample_rate)
        prepared_hits.append((event, sample_path, onset_sample, hit_audio))

    rendered_hits = [
        (event, sample_path, onset_sample, apply_fades(hit_audio, sample_rate))
        for event, sample_path, onset_sample, hit_audio in prepared_hits
    ]

    tail_padding_samples = max(0, int(round(tail_padding_seconds * sample_rate)))
    end_sample = max(onset + hit.size for _, _, onset, hit in rendered_hits) + tail_padding_samples
    timeline = np.zeros(end_sample, dtype=np.float32)

    labels: list[dict] = []
    for event, sample_path, onset_sample, hit_audio in rendered_hits:
        end = onset_sample + hit_audio.size
        timeline[onset_sample:end] += hit_audio * event.gain

        onset_seconds = onset_sample / sample_rate
        labels.append(
            {
                "instrument": event.instrument,
                "class": event.sample_class,
                "timestamp": round(onset_seconds, 6),
                "onset_sample": onset_sample,
                "duration": round(hit_audio.size / sample_rate, 6),
                "gain": round(event.gain, 3),
                "timing_offset_seconds": round(event.timing_offset_seconds, 6),
                "source_wav": str(sample_path.relative_to(PROJECT_ROOT)),
            }
        )

    return RenderedPhrase(
        audio=peak_normalize(timeline),
        labels=labels,
        sample_rate=sample_rate,
        bpm=bpm,
        tail_padding_seconds=tail_padding_seconds,
    )


def export_phrase(
    rendered: RenderedPhrase,
    audio_path: Path,
    labels_path: Path,
) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(audio_path, rendered.audio, rendered.sample_rate, subtype="FLOAT")

    payload = {
        "sample_rate": rendered.sample_rate,
        "bpm": rendered.bpm,
        "format": "mono_float32_wav",
        "peak": round(float(np.max(np.abs(rendered.audio))) if rendered.audio.size else 0.0, 6),
        "duration": round(rendered.audio.size / rendered.sample_rate, 6),
        "tail_padding_seconds": round(rendered.tail_padding_seconds, 6),
        "events": rendered.labels,
    }
    labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def step_to_beat(step: int, grid_steps: int = GRID_STEPS, beats_per_phrase: int = BEATS_PER_PHRASE) -> float:
    return round(step * (beats_per_phrase / grid_steps), 6)


def choose_timing_offset(rng: random.Random, step: int) -> float:
    if step in (0, 4, 8, 12, 16):
        return rng.uniform(-0.0025, 0.0025)
    return rng.uniform(*MICROTIMING_SECONDS_RANGE)


def make_event(
    sample_class: str,
    step: int,
    gain: float = 1.0,
    timing_offset_seconds: float = 0.0,
) -> HitEvent:
    return HitEvent(
        instrument=sample_class,
        sample_class=sample_class,
        beat=step_to_beat(step),
        gain=round(gain, 3),
        timing_offset_seconds=round(timing_offset_seconds, 6),
    )


def add_event(
    events_by_step: dict[int, list[HitEvent]],
    step: int,
    sample_class: str,
    gain: float,
    timing_offset_seconds: float = 0.0,
) -> None:
    step_events = events_by_step.setdefault(step, [])
    if any(event.sample_class == sample_class for event in step_events):
        return
    step_events.append(make_event(sample_class, step, gain, timing_offset_seconds))


def choose_accent_gain(rng: random.Random) -> float:
    return rng.uniform(*ACCENT_GAIN_RANGE)


def choose_ghost_gain(rng: random.Random) -> float:
    return rng.uniform(*GHOST_NOTE_GAIN_RANGE)


def choose_tom_gain(rng: random.Random, fill_progress: float) -> float:
    low, high = TOM_FILL_GAIN_RANGE
    return min(1.0, rng.uniform(low, high) + fill_progress * 0.12)


def choose_fill_tom(
    rng: random.Random,
    progress: float,
    previous_tom_index: int,
) -> int:
    target_index = min(int(progress * len(FILL_TOM_PATH)), len(FILL_TOM_PATH) - 1)

    if target_index < previous_tom_index:
        return previous_tom_index

    if target_index > previous_tom_index:
        return min(previous_tom_index + 1, target_index)

    if previous_tom_index < len(FILL_TOM_PATH) - 1 and rng.random() < 0.18 + progress * 0.22:
        return previous_tom_index + 1

    return previous_tom_index


def flatten_events(events_by_step: dict[int, list[HitEvent]]) -> list[HitEvent]:
    return [
        event
        for step in sorted(events_by_step)
        for event in sorted(events_by_step[step], key=lambda item: INSTRUMENT_CLASSES.index(item.sample_class))
    ]


def humanize_events(events: list[HitEvent], rng: random.Random) -> list[HitEvent]:
    humanized: list[HitEvent] = []
    for event in events:
        gain_variation = rng.uniform(-0.035, 0.035)
        gain = min(1.0, max(0.16, event.gain + gain_variation))
        step = int(round(event.beat / (BEATS_PER_PHRASE / GRID_STEPS)))
        humanized.append(
            HitEvent(
                instrument=event.instrument,
                sample_class=event.sample_class,
                beat=event.beat,
                gain=round(gain, 3),
                timing_offset_seconds=choose_timing_offset(rng, step),
            )
        )
    return humanized


def generate_groove(
    rng: random.Random,
    pattern: GroovePattern | None = None,
) -> dict[int, list[HitEvent]]:
    pattern = pattern or rng.choice(GROOVE_PATTERNS)
    events_by_step: dict[int, list[HitEvent]] = {}

    for step in pattern.kick_steps:
        add_event(events_by_step, step, "KD", choose_accent_gain(rng))

    for step in pattern.snare_steps:
        add_event(events_by_step, step, "SD", choose_accent_gain(rng))

    for step in pattern.optional_kick_steps:
        if rng.random() < 0.42:
            add_event(events_by_step, step, "KD", rng.uniform(0.62, 0.9))

    for step in pattern.ghost_snare_steps:
        if rng.random() < 0.32:
            add_event(events_by_step, step, "SD", choose_ghost_gain(rng))

    # A drummer often leaves space; remove one non-essential weak event sometimes.
    removable_steps = [
        step
        for step, events in events_by_step.items()
        if step not in pattern.kick_steps and step not in pattern.snare_steps and len(events) == 1
    ]
    if removable_steps and rng.random() < 0.25:
        events_by_step.pop(rng.choice(removable_steps))

    return events_by_step


def apply_groove_variations(events_by_step: dict[int, list[HitEvent]], rng: random.Random) -> None:
    if rng.random() < 0.3:
        add_event(events_by_step, rng.choice((2, 6, 10, 14)), "KD", rng.uniform(0.55, 0.78))

    if rng.random() < 0.22:
        add_event(events_by_step, rng.choice((7, 11, 15)), "SD", choose_ghost_gain(rng))

    if rng.random() < 0.18:
        add_event(events_by_step, 12, "KD", rng.uniform(0.45, 0.7))


def fill_start_step(rng: random.Random) -> int:
    if rng.random() < FINAL_TWO_BEAT_FILL_PROBABILITY:
        return rng.choice((8, 10))
    return rng.choice((12, 13))


def generate_fill(rng: random.Random, start_step: int | None = None) -> dict[int, list[HitEvent]]:
    start = start_step if start_step is not None else fill_start_step(rng)
    fill_steps = list(range(start, GRID_STEPS))
    events_by_step: dict[int, list[HitEvent]] = {}
    motif = rng.choice(FILL_MOTIFS)
    response_motif = tuple(offset for offset in motif if offset != 3 or rng.random() < 0.45)
    motif_length = 4
    previous_tom_index = 0
    last_fill_step = GRID_STEPS - 1

    for step in fill_steps:
        progress = (step - start) / max(GRID_STEPS - start - 1, 1)
        motif_position = (step - start) % motif_length
        cycle_index = (step - start) // motif_length
        active_motif = motif if cycle_index == 0 else response_motif
        repeated_motif_hit = motif_position in active_motif
        density_gate = 0.08 + progress * 0.28
        leave_space_before_resolution = step == last_fill_step and rng.random() < 0.68

        if not leave_space_before_resolution and (repeated_motif_hit or rng.random() < density_gate):
            tom_index = choose_fill_tom(rng, progress, previous_tom_index)
            previous_tom_index = tom_index
            sample_class = FILL_TOM_PATH[tom_index]
            add_event(events_by_step, step, sample_class, choose_tom_gain(rng, progress))

        if progress > 0.58 and rng.random() < 0.22:
            snare_gain = rng.uniform(0.42, 0.72) if step != last_fill_step else rng.uniform(0.58, 0.82)
            add_event(events_by_step, step, "SD", snare_gain)

    resolution = rng.choice(FILL_RESOLUTIONS)
    if resolution == "KD+SD":
        add_event(events_by_step, GRID_STEPS, "KD", choose_accent_gain(rng))
        add_event(events_by_step, GRID_STEPS, "SD", rng.uniform(0.68, 0.94))
    else:
        add_event(events_by_step, GRID_STEPS, resolution, choose_accent_gain(rng))

    if resolution == "KD" and rng.random() < 0.28:
        add_event(events_by_step, GRID_STEPS, "SD", rng.uniform(0.65, 0.92))

    return events_by_step


def assemble_phrase(
    rng: random.Random | None = None,
    fill_probability: float = FILL_PROBABILITY,
) -> list[HitEvent]:
    if rng is None:
        rng = random.Random()

    events_by_step = generate_groove(rng)
    apply_groove_variations(events_by_step, rng)

    if rng.random() < fill_probability:
        fill_events = generate_fill(rng)
        fill_start = min(fill_events)

        # Let the fill take over the ending instead of stacking unrelated groove notes.
        for step in [step for step in events_by_step if step >= fill_start]:
            events_by_step.pop(step)

        for step, events in fill_events.items():
            for event in events:
                add_event(events_by_step, step, event.sample_class, event.gain)

    return humanize_events(limit_phrase_density(flatten_events(events_by_step)), rng)


def limit_phrase_density(
    events: list[HitEvent],
    min_events: int = MIN_EVENTS_PER_PHRASE,
    max_events: int = MAX_EVENTS_PER_PHRASE,
) -> list[HitEvent]:
    if len(events) <= max_events:
        return events

    required = [
        event
        for event in events
        if (event.sample_class == "KD" and event.beat in (0.0, 2.0, 4.0))
        or (event.sample_class == "SD" and event.beat in (1.0, 3.0, 4.0))
    ]
    optional = [event for event in events if event not in required]

    def musical_priority(event: HitEvent) -> tuple[int, float, float]:
        is_fill_tom = event.sample_class.startswith("T") and event.beat >= 2.0
        is_resolution = event.beat >= 4.0 and event.sample_class in ("KD", "SD")
        is_ghost = event.sample_class == "SD" and event.gain <= 0.45
        priority = 0
        if is_resolution:
            priority += 5
        if is_fill_tom:
            priority += 4
        if event.sample_class in ("KD", "SD") and event.gain > 0.55:
            priority += 2
        if is_ghost:
            priority -= 2
        return (priority, event.beat, event.gain)

    optional = sorted(optional, key=musical_priority, reverse=True)
    kept = required + optional[: max(max_events - len(required), min_events - len(required))]
    return sorted(kept[:max_events], key=lambda event: (event.beat, INSTRUMENT_CLASSES.index(event.sample_class)))


def generate_random_phrase(
    rng: random.Random | None = None,
    min_events: int = MIN_EVENTS_PER_PHRASE,
    max_events: int = MAX_EVENTS_PER_PHRASE,
    grid_steps: int = GRID_STEPS,
    beats_per_phrase: int = BEATS_PER_PHRASE,
) -> list[HitEvent]:
    if grid_steps != GRID_STEPS or beats_per_phrase != BEATS_PER_PHRASE:
        raise ValueError("Pattern-based generation currently expects a 16-step, 4-beat phrase.")
    events = assemble_phrase(rng)
    return limit_phrase_density(events, min_events, max_events)


def generate_training_dataset(
    phrase_count: int = DEFAULT_PHRASE_COUNT,
    bpm: float = DEFAULT_BPM,
    samples_dir: Path = SAMPLES_DIR,
    output_audio_dir: Path = OUTPUT_AUDIO_DIR,
    output_labels_dir: Path = OUTPUT_LABELS_DIR,
    sample_rate: int = TARGET_SR,
    seed: int = 7,
) -> None:
    rng = random.Random(seed)

    for index in range(1, phrase_count + 1):
        events = generate_random_phrase(rng)
        rendered = render_phrase(
            events,
            bpm=bpm,
            samples_dir=samples_dir,
            sample_rate=sample_rate,
            seed=seed + index,
        )

        stem = f"phrase_{index:04d}_{int(round(bpm))}bpm"
        export_phrase(
            rendered,
            output_audio_dir / f"{stem}.wav",
            output_labels_dir / f"{stem}.json",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate procedural drum phrases.")
    parser.add_argument("--bpm", type=float, default=DEFAULT_BPM)
    parser.add_argument("--count", type=int, default=DEFAULT_PHRASE_COUNT)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_training_dataset(phrase_count=args.count, bpm=args.bpm, seed=args.seed)


if __name__ == "__main__":
    main()
