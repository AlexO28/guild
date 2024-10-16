#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:40:13 2023

@author: alexey.osipov
"""
import requests
import pandas as pd


example_line = "http://127.0.0.1:8000/recognize_and_extract?path=//home//alexey.osipov//gitlab//eco-ocr//data//&language=rus&key_phrase=%D0%A1%D0%BE%D0%B3%D0%BB%D0%B0%D1%88%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BE%20%D1%81%D0%BE%D1%82%D1%80%D1%83%D0%B4%D0%BD%D0%B8%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B5%20%D0%A1%D0%B8%D0%BD%D0%BA%D1%80%D0%B5%D1%82%D0%B8%D1%81%20%D0%92%D0%A1%D0%95%D0%93%D0%95%D0%98&file_with_clauses=%D0%BC%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0_%D0%BE%D0%B3%D0%BE%D0%B2%D0%BE%D1%80%D0%BE%D0%BA.csv"
example_line_2 = "http://127.0.0.1:8000/recognize_and_extract?path=//home//alexey.osipov//gitlab//eco-ocr//data//&language=both&key_phrase=%D0%94%D0%BE%D0%B3%D0%BE%D0%B2%D0%BE%D1%80%20%D1%81%D1%82%D1%80-%D0%BD%D0%B8%D1%8F%20%D0%9E%D1%82%D0%B2%D0%B5%D1%82%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D0%BE%D1%81%D1%82%D0%B8&file_with_clauses=%D0%BC%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0_%D0%BE%D0%B3%D0%BE%D0%B2%D0%BE%D1%80%D0%BE%D0%BA.csv"


def check_clauses(clauses, labels):
    answers = []
    for clause in clauses:
        answers.append(clause["clauseMatch"])
    assert answers == labels


def test_upload_endpoint():
    response = requests.post(
        "http://127.0.0.1:8000/upload",
        files=dict(
            file=open(
                "../data/Договор Им +СМР 2023_сценарий 1 Приложение №1.docx", "rb"
            )
        ),
    )
    assert response.status_code == 200
    content = response.json()
    clauses = content["data"]["clauses"]
    check_clauses(clauses, ["MATCHED", "MATCHED", "MATCHED", "PARTIAL_MATCHED"])
    response = requests.post(
        "http://127.0.0.1:8000/upload",
        files=dict(
            file=open(
                "../data/Договор Им +СМР 2023_сценарий 2 Приложение №2.docx", "rb"
            )
        ),
    )
    assert response.status_code == 200
    content = response.json()
    clauses = content["data"]["clauses"]
    check_clauses(clauses, ["NOT_MATCHED", "MATCHED", "MATCHED", "MATCHED"])
    response = requests.post(
        "http://127.0.0.1:8000/upload",
        files=dict(
            file=open(
                "../data/Договор Им +СМР 2023_сценарий 3 Приложение №3.docx", "rb"
            )
        ),
    )
    assert response.status_code == 200
    content = response.json()
    clauses = content["data"]["clauses"]
    check_clauses(clauses, ["MATCHED", "NOT_MATCHED", "MATCHED", "MATCHED"])


def test_download_endpoint():
    response = requests.get("http://127.0.0.1:8000/storage/test_file.pdf")
    assert response.status_code == 200


def test_check_vsegei_response():
    response = requests.get(example_line)
    response = response.json()
    assert len(response) == 5
    res = pd.DataFrame.from_dict(response[0]["text"], orient="index")
    assert len(res) == 223
    assert (
        str(res.head())
        == "                                          0\n0                                          \n1                                          \n2               СОГЛАШЕНИЕ О СОТРУДНИЧЕСТВЕ\n3                                         >\n4  г. Санкт-Петербург « ‚25 марта 2023 года"
    )
    assert (
        response[1]
        == '{"index":{"0":0,"1":0},"field":{"0":"Дата","1":"Тип договора"},"value":{"0":1679702400000,"1":"СОГЛАШЕНИЕ О СОТРУДНИЧЕСТВЕ\\n>"},"text":{"0":"г. Санкт-Петербург « ‚25 марта 2023 года","1":"СОГЛАШЕНИЕ О СОТРУДНИЧЕСТВЕ"}}'
    )
    assert (
        response[2]
        == '{"index":{"0":7,"1":0,"2":1,"3":2,"4":3,"5":4,"6":6,"7":8,"8":9,"9":5},"score":{"0":23,"1":10,"2":10,"3":10,"4":10,"5":10,"6":10,"7":10,"8":10,"9":-10},"found_word":{"0":"предоставлять информация ","1":"информация ","2":"информация ","3":"информация ","4":"информация ","5":"информация ","6":"информация ","7":"информация ","8":"информация ","9":"информация "},"clause_name":{"0":"Конфиденциальность (вариант 1)","1":"Конфиденциальность (вариант 1)","2":"Конфиденциальность (вариант 1)","3":"Конфиденциальность (вариант 1)","4":"Конфиденциальность (вариант 1)","5":"Конфиденциальность (вариант 1)","6":"Конфиденциальность (вариант 1)","7":"Конфиденциальность (вариант 1)","8":"Конфиденциальность (вариант 1)","9":"Конфиденциальность (вариант 1)"},"clause_text":{"0":"разрешено предоставлять информацию","1":"разрешено предоставлять информацию","2":"разрешено предоставлять информацию","3":"разрешено предоставлять информацию","4":"разрешено предоставлять информацию","5":"разрешено предоставлять информацию","6":"разрешено предоставлять информацию","7":"разрешено предоставлять информацию","8":"разрешено предоставлять информацию","9":"разрешено предоставлять информацию"},"text":{"0":"целей Соглашения. Стороны вправе предоставлять конфиденциальную информацию","1":"® Обмен информацией о деятельности Сторон с целью подготовки и","2":"® Содействие распространению информации по представляющим взаимный","3":"4.1. Стороны признают, что информация, полученная Сторонами друг от","4":"друга при реализации настоящего Соглашения, является информацией","5":"конфиденциального характера (конфиденциальная информация).","6":"4.3. Конфиденциальная информация может быть предоставлена третьим","7":"применяется, если информация:","8":"деятельности, Стороны вправе распространять информацию об установлении","9":"4.2. Конфиденциальная информация не может быть раскрыта третьим лицам"}}'
    )
    res = pd.DataFrame.from_dict(response[3], orient="index")
    assert len(res) == 3
    assert (
        str(res)
        == "                        0  ...        47\ntext      Санкт-Петербург  ...    ЕЕ  Жк\ntype                  LOC  ...       ORG\nraw_text  Санкт-Петербург  ...  ЕЕ\\n\\nЖк\n\n[3 rows x 48 columns]"
    )
    res = pd.DataFrame.from_dict(response[4], orient="index")
    assert len(res) == 3


def test_check_billingual_response():
    response = requests.get(example_line_2)
    response = response.json()
    assert len(response) == 5
    res = pd.DataFrame.from_dict(response[0]["text"], orient="index")
    assert len(res) == 1383
    assert (
        str(res.head())
        == "                                                   0\n0    Договор Страхования гражданской ответственности\n1                  NeOK06-210000876 / ОК50-200004393\n2                                                   \n3                               г. Москва 15.12.2020\n4  АО СК «Альянс», именуемое в дальнейшем “Страхо..."
    )
    assert (
        response[1]
        == '{"index":{"0":0,"1":0,"2":0,"3":2},"field":{"0":"Дата","1":"Тип договора","2":"Сумма","3":"Сумма"},"value":{"0":1607990400000,"1":"Third party Civil liability insurance\\nContract Ne OK06-210000876 \\/ OK50-200004393","2":"770 000 000.00 ","3":"770 000 000.00 "},"text":{"0":"г. Москва 15.12.2020","1":"Third party Civil liability insurance","2":"размере 770 000 000.00 (Семьсот семьдесят миллионов и","3":"amount of 770 000 000.00 (Seven hundred and seventy"}}'
    )
    assert (
        response[2]
        == '{"index":{"0":0,"1":7,"2":1,"3":2,"4":3,"5":4,"6":5,"7":6,"8":9,"9":8},"score":{"0":13,"1":13,"2":10,"3":10,"4":10,"5":10,"6":10,"7":10,"8":10,"9":9},"found_word":{"0":"предоставлять ","1":"предоставлять ","2":"информация ","3":"информация ","4":"информация ","5":"информация ","6":"информация ","7":"информация ","8":"информация ","9":"разрешить "},"clause_name":{"0":"Конфиденциальность (вариант 1)","1":"Конфиденциальность (вариант 1)","2":"Конфиденциальность (вариант 1)","3":"Конфиденциальность (вариант 1)","4":"Конфиденциальность (вариант 1)","5":"Конфиденциальность (вариант 1)","6":"Конфиденциальность (вариант 1)","7":"Конфиденциальность (вариант 1)","8":"Конфиденциальность (вариант 1)","9":"Конфиденциальность (вариант 1)"},"clause_text":{"0":"разрешено предоставлять информацию","1":"разрешено предоставлять информацию","2":"разрешено предоставлять информацию","3":"разрешено предоставлять информацию","4":"разрешено предоставлять информацию","5":"разрешено предоставлять информацию","6":"разрешено предоставлять информацию","7":"разрешено предоставлять информацию","8":"разрешено предоставлять информацию","9":"разрешено предоставлять информацию"},"text":{"0":"предоставляющего опасность для жизни, здоровья и","1":"предоставляется или доступна застрахованному лицу или её","2":"личной информации \\/ личных данных или конфиденциальной","3":"информации (кроме информации, которая законно доступна в","4":"такая информация, которая была об","5":"личную информацию \\/ персональные данные и \\/ или","6":"конфиденциальную информацию (кроме информации,","7":"широкой публике, если такая информация, которая была","8":"10.3. Информация об адресах офисов Страховщика, в","9":"Зеландию, Пуэрто Рико), где это разрешено на законных"}}'
    )
    res = pd.DataFrame.from_dict(response[3], orient="index")
    assert len(res) == 3
    assert (
        str(res)
        == "                            0                 1  ...       745       746\ntext      Договор Страхования  NeOK06-210000876  ...         9         9\ntype                   PERSON              DATE  ...  CARDINAL  CARDINAL\nraw_text  Договор Страхования  NeOK06-210000876  ...         9         9\n\n[3 rows x 747 columns]"
    )
    res = pd.DataFrame.from_dict(response[4], orient="index")
    assert len(res) == 3
