import requests
import time
import bs4


class WebScrap:
    def __init__(self, base_url):
        self.base_url = base_url

    def request_url(self):
        time.sleep(2)
        response = requests.get(self.base_url)
        html = response.text
        return html

    def parse_html(self, to_parse):
        soup = bs4.BeautifulSoup(to_parse, 'html.parser')
        return soup

    def extract_article_info(self):
        response = requests.get(self.base_url)
        article_html = response.text
        soup = self.parse_html(article_html)

        article_link = soup.findAll("img", class_ ="row")

        print(article_link)
        list_of_titles = []
        for i in article_link:
            list_of_titles.append(i.get("alt"))
        return list_of_titles

    def extract_all_images(self):
        html = self.request_url()
        soup = self.parse_html(html)
        article_link = soup.findAll("img", class_="row")
        print(article_link)
        list_of_images = []
        for i in article_link:
            list_of_images.append(i.get("data-src"))
        return list_of_images

    def find_new_link(self):
        html = self.request_url()
        soup = self.parse_html(html)
        article_link = soup.findAll("img", class_="row")
        print(article_link)


if __name__ == "__main__":
    url = "https://zeenews.india.com/latest-news"
    sp = WebScrap(url)

    image_url = sp.extract_all_images()
    print(image_url)
    title_all = sp.extract_article_info()
    print(title_all)
    sp.find_new_link()
