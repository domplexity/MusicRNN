import mido
from mido import Message, MetaMessage, MidiTrack, MidiFile
import numpy as np
import os
import argparse
from collections import namedtuple


NoteDownEvent = namedtuple("NoteDownEvent", ["note"])
NoteUpEvent = namedtuple("NoteUpEvent", ["note"])
DtEvent = namedtuple("DtEvent", ["dt"])
VelocityEvent = namedtuple("VelocityEvent", ["vel"])

# translation between numbers and events
all_events = [DtEvent(round(float(i)*0.01, 2)) for i in range(101)]+\
             [NoteDownEvent(i) for i in range(21,109)]+\
             [NoteUpEvent(i) for i in range(21, 109)] +\
             [VelocityEvent(16*i+8) for i in range(0, 8)]
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
    In particular, this class was written with the Piano Midi dataset 
    (http://www.piano-midi.de/) in mind.
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

    def discretize_velocity(self, velocity):
        return (velocity//16)*16 + 8

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
        :param num: Track number.
        :return: list of events 
        """
        events = []

        # make sure that track exists
        if num < len(self.midi_file.tracks):
            track = self.midi_file.tracks[num]#[:90] #debugging
        else:
            return events

        absolute_time_sec = 0
        absolute_time_tick = 0
        dt = 0
        for message in track:
            absolute_time_tick += message.time
            dt += mido.tick2second(message.time, self.midi_file.ticks_per_beat, self.get_tempo_at_tick(absolute_time_tick))
            absolute_time_sec += dt

            if not message.is_meta:
                # some midi files decode note_up as note_on with vel=0
                if message.type == "note_on" and message.velocity > 0:
                    if dt > 0:
                        events.append(DtEvent(dt=self.discretize_time(dt)))
                        dt = 0
                    events.append(VelocityEvent(self.discretize_velocity(message.velocity)))
                    events.append(NoteDownEvent(message.note))

                if message.type == "note_up" or (message.type == "note_on" and message.velocity == 0):
                    if dt > 0:
                        events.append(DtEvent(dt=self.discretize_time(dt)))
                        dt = 0
                    events.append(NoteUpEvent(message.note))

        return events

    def read_all_tracks(self):
        """
        Reads all tracks from midi and concatinates them
        :return: list of events of all tracks
        """
        return self.read_track(0) + self.read_track(1) + self.read_track(2)

    def write_tracks(events):
        """
        Creates midi file from event list
        :param events: midi events as a list of NoteUpEvent, NoteDownEvent, DtEvent
        :return: MidiFile with the track 
        """
        mid = MidiFile(ticks_per_beat=384) #480

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
        velocity = 64
        for event in events:
            if isinstance(event, DtEvent):
                last_dt += event.dt

            if isinstance(event, VelocityEvent):
                velocity = event.vel

            if isinstance(event, NoteDownEvent):
                delta_ticks = int(mido.second2tick(last_dt, mid.ticks_per_beat, tempo))
                mel_track.append(Message("note_on", note=event.note, time=delta_ticks, velocity=velocity))
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

        if '.mid' in filename.lower():
            filepath = args.directory + "/midi/" + filename
            events.extend(MidiFileParser(filepath).read_all_tracks())
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

        # split in train and test set
        total_length = len(total_corpus)
        train_set = total_corpus[:-total_length // 10]
        test_set = total_corpus[-total_length // 10:]

        # save test and train sets
        np.save(args.directory + "/tensors/corpus.npy", train_set)
        np.save(args.directory + "/tensors/corpus_test.npy", test_set)
    else:
        numbers = np.load(args.directory + "/tensors/corpus.npy")
        write_events(numbers, "test.mid")



