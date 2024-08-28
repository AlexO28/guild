#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:52:35 2023

@author: alexey.osipov
"""
from shiny import ui


app_ui = ui.page_fluid(
    ui.panel_title("Каков ущерб от гроз?"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select(
                "frequency",
                "Частота",
                choices=[
                    "все",
                    "до 100",
                    "от 100 до 200",
                    "от 200 до 300",
                    "от 300 до 400",
                    "от 400",
                ],
                selected="все",
            ),
            ui.input_select(
                "number_of_lightnings",
                "Количество молний",
                choices=[
                    "все",
                    "до 600",
                    "от 600 до 800",
                    "от 800 до 1000",
                    "от 1000 до 1200",
                    "от 1200",
                ],
                selected="все",
            ),
            ui.input_select(
                "wind_speed",
                "Скорость ветра (в м/сек)",
                choices=["все", "до 25", "от 25 до 26", "от 26 до 28", "от 28"],
                selected="все",
            ),
            ui.input_select(
                "probability",
                "вероятность",
                choices=[
                    "все",
                    "до 0.05",
                    "от 0.05 до 0.1",
                    "от 0.1 до 0.15",
                    "от 0.15",
                ],
                selected="все",
            ),
        ),
        ui.panel_main(
            ui.navset_tab(
                ui.nav("Графики", ui.output_plot("plot_thunderstorms")),
                ui.nav("Обзор датасета", ui.output_data_frame("table_thunderstorms")),
            )
        ),
    ),
)
