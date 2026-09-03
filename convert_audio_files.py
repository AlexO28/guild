# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 21:55:30 2026

@author: Семья
"""
import os
from pydub import AudioSegment


def convert_all_audio_files(input_folder, output_folder):
    files = os.listdir(input_folder)
    for filename in files:
        if ".null" in filename:
            audio = AudioSegment.from_file(input_folder + filename)
            audio.export(
                output_folder + filename.split(".")[0] + ".mp3",
                format="mp3",
                bitrate="192k",
            )
