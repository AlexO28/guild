#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:41:57 2023

@author: alexey.osipov
"""
from mimetypes import guess_type
from os.path import isfile
import uuid

import aiofiles
import pandas as pd
from starlette.responses import Response

from image_recognition import recognize_raw_images_from_document, save_texts_to_pdf
from clauses import extract_clauses, extract_entities_in_table
from text_analysis import extract_statistics_on_entities
from misprints import extract_misprints
from engine import (
    process_doc_file,
    uglify,
    form_outputs,
    get_processed_clause_matrix,
    update_clause_matrix,
)
from reporting import form_report, change_from_pdf_to_csv
from tuning import find_optimal_parameters, reset_parameters

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://ocr-sync.syncretis.ru:3000",
    "http://ocr-sync.syncretis.com:3000",
]
methods = ["GET", "POST", "PUT", "DELETE"]
allow_credentials = True
allow_headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=methods,
    allow_credentials=allow_credentials,
    allow_headers=allow_headers,
)


@app.post("/upload")
async def process_extracted_file(file: UploadFile):
    temp_filename = str(uuid.uuid4())
    input_file_path = "../temp/" + temp_filename + ".docx"
    output_file_path = "../storage/" + temp_filename + ".pdf"

    async with aiofiles.open(input_file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    pretty_outputs, res_dict = process_doc_file(input_file_path, output_file_path)
    ugly_outputs = uglify(pretty_outputs)
    output_report_path = change_from_pdf_to_csv(output_file_path)
    form_report(pretty_outputs, output_report_path)
    outputs = form_outputs(output_file_path, output_report_path, ugly_outputs, res_dict)

    return outputs


@app.get("/storage/{filename}")
async def get_file(filename):
    filename = "../storage/" + filename

    if not isfile(filename):
        return Response(status_code=404)

    with open(filename, "rb") as f:
        content = f.read()

    content_type, _ = guess_type(filename)

    print(content_type)
    return Response(content, media_type=content_type)


@app.get("/get_clauses/")
def get_clauses():
    clauses = get_processed_clause_matrix()
    return clauses


@app.put("/update_clauses")
def update_clauses(clauses):
    update_clause_matrix(clauses)


@app.put("/tune")
async def tune_parameters(file: UploadFile):
    temp_filename = str(uuid.uuid4())
    input_file_path = "../temp/" + temp_filename + ".csv"

    async with aiofiles.open(input_file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    parameters = find_optimal_parameters(input_file_path)
    return parameters


@app.put("/reset_to_default_parameters")
def reset_to_default_parameters():
    parameters = reset_parameters()
    return parameters


@app.get("/recognize_and_extract")
def recognize_and_extract(
    path="/home/alexey.osipov/gitlab/eco-ocr/data/",
    key_phrase="Соглашение о сотрудничестве Синкретис ВСЕГЕИ",
    language="rus",
    file_with_clauses="матрица_оговорок.csv",
):
    parsed_text = recognize_raw_images_from_document(path, key_phrase, language)
    if language == "both":
        language = "eng"
    parsed_text = parsed_text.sort_values(by=["page_number", "position_number"])
    texts = parsed_text["text"].values.tolist()
    entities = extract_entities_in_table(texts, language)
    clauses = extract_clauses(texts, path + file_with_clauses)
    standard_entities, statistics = extract_statistics_on_entities(texts, language)
    misprints = extract_misprints(texts)
    texts = pd.DataFrame(texts, columns=["text"])
    save_texts_to_pdf(texts, path + "/output_" + key_phrase + ".pdf", font=10)
    entities.reset_index(inplace=True)
    clauses.reset_index(inplace=True)
    res = [
        texts,
        entities.to_json(force_ascii=False),
        clauses.to_json(force_ascii=False),
        standard_entities,
        misprints,
    ]
    return res
