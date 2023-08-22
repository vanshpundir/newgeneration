from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from entity.news import News
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
news = News()

# Initialize WebDriver using WebDriver Manager
chromedriver_path = ChromeDriverManager().install()
driver = webdriver.Chrome()

# Load the page
driver.get('https://zeenews.india.com/latest-news')

# Wait for page to load
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.container.catergory-section-container')))

# Find the parent container element
container_element = driver.find_element(By.CSS_SELECTOR, '.container.catergory-section-container')

# Find all <a> elements within the container
a_elements = container_element.find_elements(By.CSS_SELECTOR, 'a')

# Extract href attributes and text content from each <a> element
for a_element in a_elements:
    href = a_element.get_attribute('href')
    text = a_element.text.strip()
    if href and text:
        news.news_links.append(href)
        news.title.append(text)

# Close the WebDriver
driver.quit()

# Initialize WebDriver using WebDriver Manager
driver = webdriver.Chrome()

for i in news.news_links:
    # Load the page
    driver.get(i)

    # Wait for the page to load
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.article_content')))

    # Find the <div> element with the class "article_description" and the specific ID "fullArticle"
    description_element = driver.find_element(By.CSS_SELECTOR, 'div.article_description#fullArticle')

    # Extract the text content of the description element
    description_text = description_element.text
    news.description.append(description_text)

    # Find the <img> element within the <div> element with the class "article_content article_image"
    image_element = driver.find_element(By.CSS_SELECTOR, '.article_content.article_image img')

    # Extract the "src" attribute of the image element
    image_url = image_element.get_attribute('src')
    news.image.append(image_url)

    # Close the WebDriver
    driver.quit()

pd.DataFrame({"links":news.news_links, "description":news.description,"image":news.image})
# Print extracted data
print("News Links:", news.news_links)
print("Description:", news.description)
print("Image:", news.image)
