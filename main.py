# Imports
import time
import datetime
import pytz
import logging
import re
import uuid  # noqa
import warnings
from logging import config

import cloudscraper
import numpy as np
import pandas as pd
import requests  # noqa
import yaml
from bs4 import BeautifulSoup
from googlesearch import search
from langdetect import detect
from tqdm import tqdm
import xlsxwriter


warnings.filterwarnings("ignore", category=Warning)


with open("config.yml", "r") as f:
    cfg = yaml.safe_load(f)
# config.dictConfig(cfg["logging"])

# df_path = cfg["data"]["input_data"]
df_path = r"C:\Users\Sachin.Pal\Desktop\fl\data\input_data\input.xlsx"

# Constants
chunk_size = cfg["constants"]["chunk_size"]
input_keywords = cfg["constants"]["keywords"]
create_cols = cfg["constants"]["create_columns"]
output_cols = cfg["constants"]["output_columns"]
key = cfg["constants"]["key"]
endpoint = cfg["constants"]["endpoint"]
duration = cfg["constants"]["pause"]

# Functions
def SplitDataFrame(chunk_size, df):
    df_list = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]  # noqa
    return df_list


def azure_translator(description):


    location = "westeurope"

    path = '/translator/text/v3.0/translate?'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'to': 'en'
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{
        'text': description
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    return response[0]['translations'][0]['text']


def SplitDataFrame(chunk_size, df):
    df_list = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]  # noqa
    return df_list



def google_search(query, url_set, results_needed=15, pause=duration):
    try:
        urls = []
        for j in search(query, stop=20, pause=duration):
            urls.append(j)
            if len(urls) >= results_needed:
                break

    except Exception as e:
        urls = []
        logging.info(f"Exception can't query search : {e}")

    return urls


def scrape_url(url):
    try:
        scraper = cloudscraper.create_scraper(
            delay=6,
            browser={
                "custom": "ScraperBot/1.0",
            },
        )

        req = scraper.get(url)
        soup = BeautifulSoup(req.content, "html.parser")

        text = soup.get_text(separator=" ", strip=True)

    except:
        text =  "Error in scrape_url function"
    return text


def find_match(text, keywords):
    matching_keywords = []
    for search_string in keywords:
        if search_string.lower() in text.lower():
            matching_keywords.append(search_string.lower())

    return matching_keywords


def get_metadata(search_word, input_keywords):
    """
    Given search word,input keywords return the urls,matched keywords,descriptions
    """
    urllist = []
    keywords = []
    descriptions = []
    eng_descriptions = []

    # print(f"Searching EAN number : {search_word}")
    logging.info(f"Searching EAN number : {search_word}")

    urls = google_search(search_word, url_set)

    required_urls = 3

    for url in urls:
        if not required_urls:
            break
        scraped_text = scrape_url(url)

        if scraped_text == "Error in scrape_url function":
            continue

        else:
            required_urls -= 1
            url_set.add(url)
            ean_ = search_word.split("EAN ")
            if len(ean_)>1:
                ean = ean_[1]
            else:
                ean = ""
            scraped_text = scraped_text.replace("\n", " ").strip("")

            ean_index = scraped_text.rfind(ean)
            description = scraped_text[ean_index - 500 : ean_index + 200].replace("  ", "")  # noqa

            try:
              lang = detect(scraped_text)
            except:
              break

            # eng_description = description if lang == "en" else azure_translator(description)
            eng_description = description

            matching_keywords = find_match(eng_description, input_keywords)

            urllist.append(url)
            keywords.append(matching_keywords)
            descriptions.append(description)

            eng_descriptions.append(eng_description)

    return urllist, keywords, descriptions, eng_descriptions


def SearchDataFrame(df):
    for column in create_cols:
        df[column] = ""
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Reading Data"):
        ean = row["ID_EAN"]
        brand = row["BRAND_NAME"]
        manuf_id = row["ID_MANUFACTURER"]

        if isinstance(ean, (int, float, np.int32, np.int64, np.float32, np.float64)) and ean!="":
            query = "EAN " + str(int(ean))

            urls, keywords, descriptions, eng_descriptions = get_metadata(query, input_keywords)

            matched_keywords = " ".join([item for sublist in keywords for item in sublist])

            df.at[index, "url_1"] = urls[0] if len(urls) > 0 else ""
            logging.info(f"url 1 : {urls[0] if len(urls) > 0 else ''}")

            df.at[index, "url_2"] = urls[1] if len(urls) > 1 else ""
            logging.info(f"url 2 : {urls[1] if len(urls) > 1 else ''}")

            df.at[index, "url_3"] = urls[2] if len(urls) > 2 else ""
            logging.info(f"url 3 : {urls[2] if len(urls) > 2 else ''}")

            df.at[index, "keywords_1"] = keywords[0] if len(keywords) > 0 else ""
            logging.info(f"keywords 1 : {keywords[0] if len(keywords) > 0 else ''}")

            df.at[index, "keywords_2"] = keywords[1] if len(keywords) > 1 else ""
            logging.info(f"keywords 2 : {keywords[1] if len(keywords) > 1 else ''}")

            df.at[index, "keywords_3"] = keywords[2] if len(keywords) > 2 else ""
            logging.info(f"keywords 3 : {keywords[2] if len(keywords) > 2 else ''}")

            # Description 1
            df.at[index, "description_1"] = descriptions[0] if len(descriptions) > 0 else ""
            logging.info(f"description 1 : {descriptions[0] if len(descriptions) > 0 else ''}")

            # Description Eng 1
            df.at[index, "description_eng_1"] = (
                eng_descriptions[0] if len(eng_descriptions) > 0 else ""
            )
            logging.info(
                f"eng description 1 : {eng_descriptions[0] if len(eng_descriptions) > 0 else ''}"
            )

            # Description 2
            df.at[index, "description_2"] = descriptions[1] if len(descriptions) > 1 else ""
            logging.info(f"description 2 : {descriptions[1] if len(descriptions) > 1 else ''}")

            # Description Eng 2
            df.at[index, "description_eng_2"] = (
                eng_descriptions[1] if len(eng_descriptions) > 1 else ""
            )
            logging.info(
                f"eng description 2 : {eng_descriptions[1] if len(eng_descriptions) > 1 else ''}"
            )

            # Description 3

            df.at[index, "description_3"] = descriptions[2] if len(descriptions) > 2 else ""
            logging.info(f"description 3 : {descriptions[2] if len(descriptions) > 2 else ''}")

            # Description Eng 3

            df.at[index, "description_eng_3"] = (
                eng_descriptions[2] if len(eng_descriptions) > 2 else ""
            )
            logging.info(
                f"eng description 3 : {eng_descriptions[2] if len(eng_descriptions) > 2 else ''}"
            )

        elif brand!="" and manuf_id!="":
            query = str(brand) + " " + str(manuf_id)
            query = query.strip(" ")
            urls, keywords, descriptions, eng_descriptions = get_metadata(query, input_keywords)

            matched_keywords = " ".join([item for sublist in keywords for item in sublist])

            df.at[index, "url_1"] = urls[0] if len(urls) > 0 else ""
            logging.info(f"url 1 : {urls[0] if len(urls) > 0 else ''}")

            df.at[index, "url_2"] = urls[1] if len(urls) > 1 else ""
            logging.info(f"url 2 : {urls[1] if len(urls) > 1 else ''}")

            df.at[index, "url_3"] = urls[2] if len(urls) > 2 else ""
            logging.info(f"url 3 : {urls[2] if len(urls) > 2 else ''}")

            df.at[index, "keywords_1"] = keywords[0] if len(keywords) > 0 else ""
            logging.info(f"keywords 1 : {keywords[0] if len(keywords) > 0 else ''}")

            df.at[index, "keywords_2"] = keywords[1] if len(keywords) > 1 else ""
            logging.info(f"keywords 2 : {keywords[1] if len(keywords) > 1 else ''}")

            df.at[index, "keywords_3"] = keywords[2] if len(keywords) > 2 else ""
            logging.info(f"keywords 3 : {keywords[2] if len(keywords) > 2 else ''}")

            # Description 1
            df.at[index, "description_1"] = descriptions[0] if len(descriptions) > 0 else ""
            logging.info(f"description 1 : {descriptions[0] if len(descriptions) > 0 else ''}")

            # Description Eng 1
            df.at[index, "description_eng_1"] = (
                eng_descriptions[0] if len(eng_descriptions) > 0 else ""
            )
            logging.info(
                f"eng description 1 : {eng_descriptions[0] if len(eng_descriptions) > 0 else ''}"
            )

            # Description 2
            df.at[index, "description_2"] = descriptions[1] if len(descriptions) > 1 else ""
            logging.info(f"description 2 : {descriptions[1] if len(descriptions) > 1 else ''}")

            # Description Eng 2
            df.at[index, "description_eng_2"] = (
                eng_descriptions[1] if len(eng_descriptions) > 1 else ""
            )
            logging.info(
                f"eng description 2 : {eng_descriptions[1] if len(eng_descriptions) > 1 else ''}"
            )

            # Description 3

            df.at[index, "description_3"] = descriptions[2] if len(descriptions) > 2 else ""
            logging.info(f"description 3 : {descriptions[2] if len(descriptions) > 2 else ''}")

            # Description Eng 3

            df.at[index, "description_eng_3"] = (
                eng_descriptions[2] if len(eng_descriptions) > 2 else ""
            )
            logging.info(
                f"eng description 3 : {eng_descriptions[2] if len(eng_descriptions) > 2 else ''}"
            )

        # print(f"matched Keywords : {matched_keywords}")

        else:
            matched_keywords = ""

        if not matched_keywords:
            df.at[index, "Tyre"] = "No"
            df.at[index, "Type"] = ""

        else:
            df.at[index, "Tyre"] = (
                "Yes" if ("tyre" in matched_keywords or "tire" in matched_keywords) else "No"
            )

            df.at[index, "Type"] = (
                "Agricultural"
                if any(word in matched_keywords for word in ["farm", "tractor"])
                else "Motorcycle"
                if any(word in matched_keywords for word in ["motorcycle","bike"])
                else "Truck"
                if any(word in matched_keywords for word in ["truck", "light truck"])
                else "Car"
                if any(
                    word in matched_keywords
                    for word in ["passenger car", "car", "sedan", "SUV", "suv","van", "bus","summer tyres","all-season tyres"]
                )
                else ""
            )

    df1 = df[output_cols]

    return df1


if __name__ == "__main__":
    start = time.time()
    url_set = set()
    # Load the dataframe
    flag=0
    try:
        df = pd.read_excel(df_path)
        df = df.fillna("")

    except:
        flag=1
        print("Input file not found")

    if flag==0:
      input_rows = df.shape[0]
      print(f"Total {input_rows} rows of data are there in the input file")
      print("----------------------------------------------")

      # Split dataframe if num_rows > chunk_size
      df_list = SplitDataFrame(chunk_size, df)

      df_output_list = []

      for dataframe in df_list:
          df_output_list.append(SearchDataFrame(dataframe))

      df_output = pd.concat(df_output_list)
      unique_url_list = list(url_set)
      column_name = ["Unique URL's"]
      url_df = pd.DataFrame(unique_url_list,columns=column_name)
      end = time.time()

      time_diff = end - start
      hrs = int(time_diff//3600)
      mins = int((time_diff%3600)//60)
      sec = int((time_diff%3600)%60)
      print("----------------------------------------------")
      print(f"The program ran for {hrs} hours, {mins} minutes, {sec} seconds")

      rows = df_output.query('url_1 != ""').shape[0]
      print(f"Total {rows} rows are filled during this time")
      print("----------------------------------------------")

      # Get the current date and time in IST
      now = datetime.datetime.utcnow()

      ist = pytz.timezone('Asia/Kolkata')
      ist_now = now.replace(tzinfo=pytz.utc).astimezone(ist)

      timestamp = ist_now.strftime("%d-%m-%Y_%H-%M-%S")

      # output_path = cfg["data"]["output_data"] + f"output_{timestamp}.xlsx"
      output_path = "C:/Users/Sachin.Pal/Desktop/fl/data/output_data/" + f"output_{timestamp}.xlsx"
      # url_path = cfg["data"]["output_data"] + f"unique_url_list_{timestamp}.xlsx"
      url_path = r"C:\Users\Sachin.Pal\Desktop\fl\data\output_data" + f"unique_url_list_{timestamp}.xlsx"
      df_output.to_excel(output_path, index=False,engine= 'xlsxwriter' )
      print("Final output excel file generated")
      print("----------------------------------------------")
      #url_df.to_excel(url_path, index=False)
      #print(f"URLs list generated with {len(url_df)} unique URLs")
      print("----------------------------------------------")
