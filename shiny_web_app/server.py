#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:10:35 2023

@author: alexey.osipov
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shiny import render, reactive
from prepare_synthetic_dataset import prepare_synthetic_dataset


def app_server(input, output, session):
    @output
    @render.plot
    def plot_thunderstorms():
        tab = prepare_synthetic_dataset(
            input.frequency(),
            input.number_of_lightnings(),
            input.wind_speed(),
            input.probability(),
        )
        tab = tab[["Частота", "Скорость ветра", "Количество молний"]]
        pd.plotting.scatter_matrix(tab, hist_kwds={"bins": 100})
        return plt.tight_layout()

    @output
    @render.data_frame
    def table_thunderstorms():
        tab = prepare_synthetic_dataset(
            input.frequency(),
            input.number_of_lightnings(),
            input.wind_speed(),
            input.probability(),
        )
        return tab
