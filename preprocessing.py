import mido
from mido import Message, MetaMessage, MidiTrack, MidiFile
import numpy as np
import os
import argparse
import pickle
from collections import namedtuple


NoteDownEvent = namedtuple("NoteDownEvent", ["note"])
NoteUpEvent = namedtuple("NoteUpEvent", ["note"])
DtEvent = namedtuple("DtEvent", ["dt"])

# translation between numbers and events
all_events = [DtEvent(round(float(i)*0.01, 2)) for i in range(101)]+[NoteDownEvent(i) for i in range(21,109)]+[NoteUpEvent(i) for i in range(21,109)]
dic_numbers_to_events = {i: event for i, event in enumerate(all_events)}
dic_events_to_numbers = {repr(dic_numbers_to_events[key]): key for key in dic_numbers_to_events}


def events_to_number(events):
    numbers = []

    for event in events:
        if repr(event) in dic_events_to_numbers:
            numbers.append(dic_events_to_numbers[repr(event)])
        else:
            raise IndexError("Event %s not in dictionary" % event)
    return numbers


def numbers_to_events(numbers):
    return [dic_numbers_to_events[num] for num in numbers]


class MidiFileParser:
    """
    This class parses midi files. It expects files with the first track being the tempo track and the other being tracks 
    for piano. 
    In particular, this class was written with the Yahama ePiano Competition dataset 
    (http://www.piano-e-competition.com) in mind.
    """
    def __init__(self, filename):
        self.midi_file = MidiFile(filename)
        self.tempi = self.read_tempi()

    def read_tempi(self):
        absolute_time = 0
        tempi = {}

        tempo_track = self.midi_file.tracks[0]

        for message in tempo_track:
            if message.type == 'set_tempo':
                absolute_time += message.time
                tempi[absolute_time] = message.tempo

        return tempi

    def discretize_time(self, time):
        if time > 1.0:
            return 1.0

        return round(time, 2)

    def get_tempo_at_tick(self, abs_tick):
        """
        Converts abs_tick from ticks in seconds. This is non-trivial as the conversion depends on the tempo track of the 
        midi file.
        :param abs_tick: absolute time in ticks
        :return: absolute time in seconds
        """
        last_tick = 0
        for tick in self.tempi:
            if tick >= abs_tick:
                return self.tempi[last_tick]
            last_tick = tick
        # if abs_tick is greater than last tempo, then return last
        return self.tempi[last_tick]

    def read_track(self, num):
        """
        Reads track from midi_file. 
        :param num: Track number. Note: In midi file this is actually track number-1 as track number=0 is tempo track 
        :return: list of events 
        """
        events = []
        track = self.midi_file.tracks[num+1]#[:90] #debugging

        absolute_time_sec = 0
        absolute_time_tick = 0
        for message in track:
            if not message.is_meta:
                absolute_time_tick += message.time
                dt = mido.tick2second(message.time, self.midi_file.ticks_per_beat, self.get_tempo_at_tick(absolute_time_tick))
                absolute_time_sec += dt

                # some midi files decode note_off as note_on with vel=0
                if message.type == "note_on" and message.velocity > 0:
                    if dt > 0:
                        events.append(DtEvent(dt=self.discretize_time(dt)))
                    events.append(NoteDownEvent(message.note))

                if message.type == "note_up" or (message.type == "note_on" and message.velocity == 0):
                    if dt > 0:
                        events.append(DtEvent(dt=self.discretize_time(dt)))
                    events.append(NoteUpEvent(message.note))

        return events

    def read_both_tracks(self):
        """
        Reads both tracks from midi and concatinates them
        :return: list of events of both tracks
        """
        return self.read_track(0) + self.read_track(1)

    def write_tracks(events):
        """
        Creates midi file from event list
        :param events: midi events as a list of NoteUpEvent, NoteDownEvent, DtEvent
        :return: MidiFile with the track 
        """
        mid = MidiFile(ticks_per_beat=480)

        # tempo track controls conversion of ticks in sec
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo = 500000
        tempo_track.append(MetaMessage("set_tempo", tempo=tempo, time=0))

        # melody track
        mel_track = MidiTrack()
        mid.tracks.append(mel_track)
        last_dt = 0
        rounding_error = 0
        for event in events:
            if isinstance(event, DtEvent):
                last_dt = event.dt

            if isinstance(event, NoteDownEvent):
                delta_ticks = int(mido.second2tick(last_dt, mid.ticks_per_beat, tempo))
                mel_track.append(Message("note_on", note=event.note, time=delta_ticks, velocity=75))
                last_dt = 0

                rounding_error += abs(delta_ticks - mido.second2tick(last_dt, mid.ticks_per_beat, tempo))

            if isinstance(event, NoteUpEvent):
                delta_ticks = int(mido.second2tick(last_dt, mid.ticks_per_beat, tempo))
                mel_track.append(Message("note_off", note=event.note, time=delta_ticks))
                last_dt = 0

                rounding_error += abs(delta_ticks - mido.second2tick(last_dt, mid.ticks_per_beat, tempo))

        print(rounding_error)
        return mid


def parse_directory(path, verbose=False):
    """
    Parses all midi files in directory path and returns a list of numbers representing them
    :param path: path to directory
    :param verbose: turns on logging information
    :return: list of integers 
    """
    events = []
    for filename in os.listdir(path):

        if verbose:
            print(filename)

        if '.mid' in filename:
            filepath = args.directory + "/midi/" + filename
            events.extend(MidiFileParser(filepath).read_both_tracks())
        else:
            if verbose:
                print('skipped file: ', filename)

    return events_to_number(events)


def write_events(numbers, filename):
    """
    Creates a midifile from numbers representing the midi events
    :param numbers: list of integers representing midi events
    :param filename: filename of midifile 
    :return: 
    """
    events = [dic_numbers_to_events[num] for num in numbers]
    file = MidiFileParser.write_tracks(events)
    file.save(filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--directory', action="store", type=str)
    argparser.add_argument('--reverse', action='store_true')
    args = argparser.parse_args()

    print("Starting to convert with directory %s" % (args.directory))

    if not args.reverse:
        total_corpus = parse_directory(args.directory + "/midi", verbose=True)
        np.save(args.directory + "/tensors/corpus.npy", total_corpus)
    else:
        numbers = np.load(args.directory + "/tensors/corpus.npy")
        write_events(numbers, "test.mid")



