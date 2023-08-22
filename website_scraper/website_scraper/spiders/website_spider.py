import scrapy
from scrapy.http import HtmlResponse
from urllib.parse import urljoin
import sqlite3

class WebsiteSpider(scrapy.Spider):
    name = "website_spider"
    allowed_domains = ["indiatoday.in"]
    start_urls = ["https://www.indiatoday.in/news.html"]

    def parse(self, response):
        # Assuming you have a list of links obtained using response.css()
        links = response.css("div .lhs__section a")

        # Get the base URL for relative URL conversion
        base_url = response.url

        # Iterate through each link and extract its information if it has an href attribute
        for link in links:
            href = link.attrib.get('href')
            if href:
                # Convert relative URL to complete URL using urljoin
                complete_href = urljoin(base_url, href)
                yield scrapy.Request(complete_href, callback=self.parse_link_page, meta={'link_text': link.css("::text").get()})

    def parse_link_page(self, response):
        # Extract the main image source
        main_image_src = response.css(".Story_associate__image__bYOH_ img::attr(src)").get()

        # Extract text from all <p> tags
        description_text = " ".join(response.css("p::text").getall())

        # Get the link text from the meta data
        link_text = response.meta.get('link_text', '')

        # Yield the extracted information
        yield {
            'link_text': link_text,
            'main_image_src': main_image_src,
            'description_text': description_text
        }

# class SQLitePipeline:
#     def __init__(self):
#         self.db_name = None
#
#     def open_spider(self, spider):
#         self.db_name = "data.db"
#         if self.db_name:
#             self.conn = sqlite3.connect(self.db_name)
#             self.cursor = self.conn.cursor()
#             self.cursor.execute('''
#                 CREATE TABLE IF NOT EXISTS scraped_data (
#                     link_text TEXT,
#                     main_image_src TEXT,
#                     description_text TEXT
#                 )
#             ''')
#
#     def close_spider(self, spider):
#         if self.db_name:
#             self.conn.commit()
#             self.conn.close()
#
#     def process_item(self, item, spider):
#         if self.db_name:
#             self.cursor.execute('''
#                 INSERT INTO scraped_data (link_text, main_image_src, description_text)
#                 VALUES (?, ?, ?)
#             ''', (item['link_text'], item['main_image_src'], item['description_text']))
#         return item
