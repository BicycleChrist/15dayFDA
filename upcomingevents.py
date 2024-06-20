from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from bs4 import BeautifulSoup

def Main():
    url = 'https://app.bpiq.com/catalyst-calendar'
    options = FirefoxOptions()
    options.add_argument('--headless')  # Run in headless mode
    service = Service()
    driver = webdriver.Firefox(service=service, options=options)

    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, 'table'))
    )
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    table = soup.find('table')  # Adjust selector if needed
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]

    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        row = [cell.get_text(strip=True) for cell in cells]
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Perform data cleanup: separate 'Company' and 'Price'
    df['Price'] = df['Company'].apply(lambda x: x.split('$')[1] if '$' in x else None)
    df['Company'] = df['Company'].apply(lambda x: x.split('$')[0] if '$' in x else x)

    # Rearrange columns so 'Price' is next to 'Company'
    columns = df.columns.tolist()
    columns.insert(1, columns.pop(columns.index('Price')))
    df = df[columns]

    # Save DataFrame to CSV
    df.to_csv('output.csv', index=False)

    print('Data has been saved to output.csv')

    driver.quit()

if __name__ == "__main__":
    Main()
