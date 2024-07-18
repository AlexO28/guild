#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:51:17 2023

@author: alexey.osipov
"""
from PIL import Image
import pytesseract
import os
import pandas as pd
from fpdf import FPDF
from clauses import extract_entities_in_table, extract_clauses


def show_raw_image(folder, filename):
    return Image.open(folder + filename)


def recognize_raw_images_from_document(folder, keyphrase, lang="rus"):
    files = os.listdir(folder)
    files = [file for file in files if (keyphrase in file) and (".png" in file)]
    if len(files) == 0:
        raise "Нет документов с данной ключевой фразой!"
    results = []
    page_numbers = []
    page_number = 0
    for file in files:
        recognized_image = recognize_raw_image(folder, file, lang=lang)
        page_number = int(
            file.replace(keyphrase, "").replace(".png", "").replace("-", "")
        )
        page_numbers.extend([page_number] * len(recognized_image))
        results.extend(recognized_image)
    parsed_text = pd.DataFrame({"text": results, "page_number": page_numbers})
    parsed_text["position_number"] = parsed_text.index
    return parsed_text


def recognize_raw_image(folder, filename, lang="rus"):
    if lang == "both":
        parsed_image = pytesseract.image_to_string(
            folder + filename, config=r"-l eng+rus"
        )
    else:
        parsed_image = pytesseract.image_to_string(folder + filename, lang=lang)
    return parsed_image.split("\n")


def save_texts_to_pdf(texts, output_file, font=10, matches={}, hack=False):
    if os.path.exists(output_file):
        os.remove(output_file)
    pdf = FPDF()
    pdf.add_page()
    MAX_LEN = 100
    for j in range(len(texts)):
        text = texts["text"][j]
        pdf.add_font("DejaVu", "", "DejaVuSansCondensed.ttf", uni=True)
        pdf.set_font("DejaVu", "", font)
        k = 0
        while k < len(text):
            textred = text[k : min(k + MAX_LEN, len(text))]
            k += MAX_LEN
            if len(matches) > 0:
                if j in matches.keys():
                    match = matches[j]
                    if match == "MATCHED":
                        pdf.set_text_color(0, 255, 0)
                    elif match == "CONTRADICTS":
                        pdf.set_text_color(255, 0, 0)
                    elif match == "PARTIAL_MATCHED":
                        pdf.set_text_color(255, 255, 0)
                    else:
                        pdf.set_text_color(0, 0, 0)
                else:
                    pdf.set_text_color(0, 0, 0)
            if hack:
                if "ДОГОВОР ОБЛИГАТОРНОГО" in text:
                    pdf.set_text_color(0, 0, 255)
                elif "01.01.2023" in text:
                    pdf.set_text_color(0, 0, 255)
            pdf.cell(0, h=5, txt=textred, ln=1)
    pdf.output(output_file)


if __name__ == "__main__":
    inputs = pd.read_csv("input_for_image_recognition.csv")
    entities_on_document = inputs["entities_on_document"].values.tolist()[0]
    parsed_text = recognize_raw_images_from_document(
        os.getcwd() + "/../data/",
        inputs["document_name"].values.tolist()[0],
        inputs["language"].values.tolist()[0],
    )
    parsed_text = parsed_text.sort_values(by=["page_number", "position_number"])
    outputs = pd.DataFrame(parsed_text["text"].values.tolist(), columns=["text"])
    if entities_on_document == False:
        outputs.to_csv("output_for_image_recognition.csv")
    else:
        language = inputs["language"].values.tolist()[0]
        if language == "both":
            language = "eng"
        entities = extract_entities_in_table(
            parsed_text["text"].values.tolist(), language
        )
        clauses = extract_clauses(parsed_text["text"].values.tolist())
        entities.to_csv("output_entities_for_image_recognition.csv")
        clauses.to_csv("output_clauses_for_image_recgnition.csv")
        outputs.to_csv("output_for_image_recognition.csv")
