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
        return self.tempi[tick]

    def read_track(self, num):
        """
        Reads track from midi_file. 
        :param num: Track number. Note: In midi file this is actually track number-1 as track number=0 is tempo track 
        :return: list of events 
        """
        events = []
        track = self.midi_file.tracks[num+1]#[:10] #debugging

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
                        events.append(DtEvent(dt=dt))
                    events.append(NoteDownEvent(message.note))

                if message.type == "note_up" or (message.type == "note_on" and message.velocity == 0):
                    if dt > 0:
                        events.append(DtEvent(dt=dt))
                    events.append(NoteUpEvent(message.note))

        return events

    def write_tracks(self, events):
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

if __name__ == "__main__":
    # for testing purposes. This will change soon.
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--directory', action="store", type=str)
    #argparser.add_argument('--reverse', action='store_true')
    #argparser.add_argument('--onehot', action='store_true')
    args = argparser.parse_args()

    print("Starting to convert with directory %s" % (args.directory))

    parser = MidiFileParser(args.directory + "/appass_1.mid")
    mel_track1 = parser.read_track(1)
    mel_track0 = parser.read_track(0)
    file = parser.write_tracks(mel_track1)
    file2 = parser.write_tracks(mel_track0)

    #file.tracks.append(file2.tracks[1])

    file.save("output.mid")

    print("Done with conversion")








