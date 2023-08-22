from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from entity.news import News


news = News()
# Initialize WebDriver using WebDriver Manager
driver = webdriver.Chrome()

driver.get('https://zeenews.india.com/latest-news')

# Wait for page to load
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.container.catergory-section-container')))

# Find the parent container element
container_element = driver.find_element(By.CSS_SELECTOR, '.container.catergory-section-container')

# Find all <a> elements within the container
a_elements = container_element.find_elements(By.CSS_SELECTOR, 'a')

# Extract href attributes and text content from each <a> element
a_data = []
for a_element in a_elements:
    href = a_element.get_attribute('href')
    text = a_element.text.strip()
    if href and text:
        a_data.append({'href': href, 'text': text})


# Print or process the extracted data
for a_item in a_data:
    news.news_links.append(a_item['text'])
    news.news_links.append(a_item['href'])


# Close the WebDriver
driver.quit()
