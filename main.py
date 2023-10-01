import subprocess
import os
import pandas as pd
from fetching_api_data.news_using_api import fetch_news

def run_scrapy_crawl():
    project_directory = "/Users/vansh/Desktop/newgeneration/newgeneration/website_scraper"
    os.chdir(project_directory)

    command = "scrapy crawl website_spider -o /Users/vansh/Desktop/newgeneration/newgeneration/data/scraped_news_new1.csv"

    try:
        subprocess.run(command, shell=True, check=True)
        print("Scrapy crawl completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def fetch_and_combine_api_data():
    urls = [
        'https://newsapi.org/v2/everything?q=tesla&from=2023-09-01&sortBy=publishedAt&apiKey=dc54d42f65ad4b1594c9617dd676dbd8',
        'https://newsapi.org/v2/everything?q=tesla&from=2023-09-01&sortBy=publishedAt&apiKey=dc54d42f65ad4b1594c9617dd676dbd8'
    ]

    all_news = []
    for url in urls:
        news = fetch_news(url)
        print(news)
        all_news.append(news)

    combined_news = {
        "link_text": [],
        "main_image_src": [],
        "description_text": []
    }

    for news in all_news:
        combined_news["link_text"].extend(news["title"])
        combined_news["main_image_src"].extend(news["image"])
        combined_news["description_text"].extend(news["description"])

    df_combined = pd.DataFrame(combined_news)
    csv_file = "/Users/vansh/Desktop/newgeneration/newgeneration/data/api_data_3.csv"
    df_combined.to_csv(csv_file, index=False)
    return csv_file

def concatenate_csv_files(csv_file1, csv_file2, output_csv):
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    concatenated_df = pd.concat([df1, df2], ignore_index=True)
    concatenated_df.to_csv(output_csv, index=False)
    print("CSV files concatenated and saved to", output_csv)

def main():
    run_scrapy_crawl()
    api_data_csv = fetch_and_combine_api_data()

    csv_file1 = "/Users/vansh/Desktop/newgeneration/newgeneration/data/scraped_news_new1.csv"
    csv_file2 = api_data_csv
    output_csv = "/Users/vansh/Desktop/newgeneration/newgeneration/data/concatenated_data.csv"

    concatenate_csv_files(csv_file1, csv_file2, output_csv)

if __name__ == "__main__":
    main()
