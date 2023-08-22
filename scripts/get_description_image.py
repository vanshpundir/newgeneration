from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from entity.news import News
from webdriver_manager.chrome import ChromeDriverManager


news = News()
chromedriver_path = ChromeDriverManager().install()

# Initialize WebDriver using WebDriver Manager
driver = webdriver.Chrome()

# Load the page
driver.get('https://zeenews.india.com/cricket/meet-chacha-cricket-pakistans-beloved-cricket-ambassador-who-is-a-big-fan-of-virat-kohli-2651895.html')  # Replace with the actual URL

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

print(news.news_links)
print(news.title)
print(news.description)
print(news.image)
# Close the WebDriver
driver.quit()
