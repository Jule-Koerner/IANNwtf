import tensorflow as tf
import numpy as np
import mido
import pretty_midi
import math
import music21


def check_time_signature(midi_path: str) -> bool:
    """Checks if song is in 4/4 time signature.

    Args:
      midi_path:
        A string containing the path of the midi file.

    Returns:
      A boolean. True if song is in 4/4 time. False if not.
    """
    mido_instance = mido.MidiFile(midi_path)

    for i, msg in enumerate(mido_instance.tracks[0]):
        if msg.type == "time_signature":
            time_sig = mido_instance.tracks[0][i]
            if time_sig.denominator != 4 or time_sig.numerator != 4:
                return False
            else:
                return True


def filter_for_track(midi_path: str, track_name: str = "PIANO"):
    """Removes all tracks of the pop MIDI file except for one.

    Args:
      midi_path:
        A string containing the path of the midi file.
      track_name:
        Track name as string of the track that should be kept.

    Returns:
      Pretty MIDI instance containing only one track
    """
    sample = pretty_midi.PrettyMIDI(midi_path)

    new_mid = pretty_midi.PrettyMIDI()
    new_track = pretty_midi.Instrument(program=42)
    new_mid.instruments.append(new_track)

    for i, inst in enumerate(sample.instruments):
        if inst.name == track_name:
            new_track.notes = inst.notes
            new_track.name = inst.name
            new_track.control_changes = inst.control_changes
            new_track.pitch_bends = inst.pitch_bends

    new_mid.write(f"{track_name}.mid")

    return new_mid


def get_PrettyMIDI(midi_path: str):
    """
    Loads MIDI file into PrettyMIDI object.
    """
    return pretty_midi.PrettyMIDI(midi_path)


def split_to_samples(mid: pretty_midi.PrettyMIDI, bars: int = 4) -> list[np.ndarray]:
    """Splits song into smaller samples.

    Uses sixteenth notes as measure for the split.
    Assumes song to be in 4/4 time signature.

    Args:
      mid:
        PrettyMIDI instance of whole song.
      bars:
        Integer defining split length by number of bars per split.

    Returns:
      A list contraining the splits. Splits are numpy arrays?
    """
    bpm = 120
    possible_pitches = 128

    quarter_length = 60 / bpm
    sixteenth_per_sec = 1 / ((quarter_length * 4) / 16)

    sample_end = mid.get_end_time()
    total_sixteenth = math.ceil(sixteenth_per_sec * sample_end)

    sixteenth_per_split = 16 * bars
    total_splits = math.floor(total_sixteenth / sixteenth_per_split)
    if total_splits == 0:
        return []

    song_as_matrix = np.zeros((total_sixteenth + 1, possible_pitches))

    for note in mid.instruments[0].notes:
        note_start = round(note.start * sixteenth_per_sec)  # note start in number of sixteenth
        note_end = round(note.end * sixteenth_per_sec)

        song_as_matrix[note_end, note.pitch] = 3  # mark note end
        song_as_matrix[note_start, note.pitch] = 1  # mark note start
        song_as_matrix[note_start + 1:note_end, note.pitch] = 2  # mark note held

    song_as_matrix = song_as_matrix[0:total_splits * sixteenth_per_split, :]
    splits = np.split(song_as_matrix, total_splits)
    splits = [s for s in splits if not np.all(s == 0)]  # only take matrices that contain notes

    return splits


def labeling(genre: str, num_labels: int) -> list:
    """Generates genre labels.

    Args:
      genre:
        A string defining the music genre. String should be written
        in lower-case. For example: "pop", "jazz", "classic".
      num_labels:
        An integer defining the number of labels needed.

    Return:
      A list containing the respective labels.
    """
    genres = {"pop": 1, "jazz": 2, "classic": 3}

    try:
        labels = [genres[genre]] * num_labels
    except:
        print("Make sure everything is written in lower-case")

    return labels


def get_ds(midi_filepaths, genre: str):
    ds = [f for f in midi_filepaths if check_time_signature(f)]
    ds = [get_PrettyMIDI(f) for f in ds]
    ds = [split_to_samples(p_midi) for p_midi in ds]
    ds = [elem for sublist in ds for elem in sublist]
    ds = tf.data.Dataset.from_tensor_slices((ds, labeling(genre, len(ds))))
    return ds


def chordify_and_save(midi_filepaths, save_path="Chordified"):
    for i, f in enumerate(midi_filepaths):
        m = music21.converter.parse(f)
        chords = m.chordify()
        chords.write('midi', fp=save_path + '/' + str(i) + '.mid')
