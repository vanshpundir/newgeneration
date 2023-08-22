import requests
import pandas as pd
import os

def fetch_news(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        if articles:
            dict_news = {
                "title": [],
                "url": [],
                "image": [],
                "description": []
            }
            for article in articles:
                dict_news["title"].append(article.get('title', 'N/A'))
                dict_news["url"].append(article.get('url', 'N/A'))
                dict_news["image"].append(article.get('urlToImage'))
                dict_news["description"].append(article.get('description', 'N/A'))
            return dict_news
        else:
            print("No articles found.")
    else:
        print("Error fetching data:", response.status_code)

def main():
    urls = [
        "https://newsapi.org/v2/everything?q=apple&from=2023-08-21&to=2023-08-21&sortBy=popularity&apiKey=dc54d42f65ad4b1594c9617dd676dbd8",
        "https://newsapi.org/v2/everything?q=apple&from=2023-08-21&to=2023-08-21&sortBy=popularity&apiKey=dc54d42f65ad4b1594c9617dd676dbd8"
    ]

    all_news = []
    for url in urls:
        news = fetch_news(url)
        all_news.append(news)

    combined_news = {
        "title": [],
        "url": [],
        "image": [],
        "description": []
    }

    for news in all_news:
        combined_news["title"].extend(news["title"])
        combined_news["url"].extend(news["url"])
        combined_news["image"].extend(news["image"])
        combined_news["description"].extend(news["description"])

    df_combined = pd.DataFrame(combined_news)
    csv_file = "data/api_data_2.csv"

    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        df_combined = pd.concat([existing_df, df_combined], ignore_index=True)

    df_combined.to_csv(csv_file, index=False)

if __name__ == "__main__":
    main()
