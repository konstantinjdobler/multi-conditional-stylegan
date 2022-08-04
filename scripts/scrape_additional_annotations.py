from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
from multiprocessing import Pool
import os
import re
import time
import lxml  # needed for beautofulsoup

"""
Scraping the Wikiarts web page for the downloaded ArtEmis images
"""


def _fix(string):
    try:
        return string.replace("\n", "").replace(":", "").strip()
    except Exception as e:
        print(e)


def _get_csv_entries(csv_file):

    pd_csv = pd.read_csv(csv_file, sep=",", header=None, names=["url"])
    print(pd_csv["url"])
    return pd_csv["url"]


def _get_html_request(property_url):
    try:
        # prevent request errors
        time.sleep(0.01)
        html_request = requests.get(property_url)
    except Exception as e:
        print(f"REQUEST ERROR: {property_url}\n\terror_desc: {e}")
        # adding flag for erroneous request calls
        html_request = -1

    return html_request


def crawl(url, multiple=[]):
    data = {}

    url_wikiart = "https://www.wikiart.org/en"
    picture_name = "_".join(url.split("/")[-2:]).replace(".jpg", "")
    property_url = f"{url_wikiart}/{picture_name.replace('_', '/')}"

    # GET request to fetch the raw HTML content
    html_request = _get_html_request(property_url)

    # cleaning up url suffixes after 404
    if html_request != -1:
        request_status_code = html_request.status_code
        if request_status_code != 200:
            clean_property_url = re.sub(r"\(\d+\)?", "", property_url)
            html_request = _get_html_request(clean_property_url)
            if html_request != -1:
                request_status_code = html_request.status_code
            else:
                print(f"REQUEST REJECTED: {picture_name}")
                return (None, f"error:{picture_name}")

        if request_status_code == 200:
            data["name"] = picture_name
            data["pic_url"] = url
            print(data["name"])

            html_content = html_request.text
            data["property_url"] = property_url

            # parse the html content with lxml (> pip install lxml)
            soup = BeautifulSoup(html_content, "lxml")

            # get properties
            properties_tree = soup.find_all("article")
            if len(properties_tree) >= 1:
                properties = properties_tree[0]

                title = properties.find("h3")
                if title is not None:
                    data["title"] = _fix(title.text)

                painter = properties.find("h5")
                if painter is not None:
                    data["painter"] = _fix(painter.text)

                listing = properties.find("ul")
                if listing is not None:
                    items = listing.find_all("li")
                    for item in items:
                        property_key = item.find("s")
                        property_value = item.find("span")
                        if property_key is not None and property_value is not None:
                            # check for multiple sub-values for current property
                            value_list = _fix(property_value.text).split(",")
                            fixed_property_key = _fix(property_key.text).lower()

                            # filter properties where len(value_list) >= 1 for at least one painting
                            if fixed_property_key in ["date", "genre", "location"]:
                                data[fixed_property_key] = _fix(property_value.text)
                            else:
                                keywords_list = []
                                for value in value_list:
                                    keywords_list.append(_fix(value))

                                data[fixed_property_key] = keywords_list

            # get keywords
            keywords_tree = soup.find("div", "tags-cheaps")
            keywords_list = []
            if keywords_tree is not None:
                html_keywords_list = keywords_tree.find_all(
                    "a", "tags-cheaps__item__ref"
                )

                for keyword in html_keywords_list:
                    keywords_list.append(_fix(keyword.text))

            data["keywords"] = keywords_list

            # return json object & None for the error
            return (data, None)

        else:
            print(f"404 NOT FOUND: {picture_name}")
            return (None, f"404:{picture_name}")

    else:
        print(f"REQUEST REJECTED: {picture_name}")
        return (None, f"error:{picture_name}")


if __name__ == "__main__":

    json_file = {}
    out_dir_stats = "./artemis-download-stats"

    csv_file = "./artemis-download-stats/all_paintings.txt"

    csv_entries = _get_csv_entries(csv_file)

    # os.cpu_count() equals 128 on gpu server -> triggers too many requests simultaneously
    # NOTE: you can increase the number of threads but you might encounter request errors (connetion reset by peer)
    with Pool(1) as p:
        json_array = p.map(crawl, csv_entries)

    json_list = list(filter(None, [tup[0] for tup in json_array]))
    json_file["entries"] = json_list
    not_found_list = list(filter(None, [tup[1] for tup in json_array]))
    with open(out_dir_stats + "/enhanced_annotations.json", "w") as outfile:
        json.dump(json_file, outfile, indent=2)

    print(f"number of successfully scraped painting links: {len(json_list)}	")

    with open(
        out_dir_stats + "/additional_annotation_not_found_list.txt", "a+"
    ) as outfile:
        for name in not_found_list:
            outfile.write(f"{name}\n")

    print(f"number of defective painting links: {len(not_found_list)}")
