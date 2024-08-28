#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:50:51 2023

@author: alexey.osipov
"""
from shiny import App
from ui import app_ui
from server import app_server


app = App(app_ui, app_server)
