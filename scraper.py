from bs4 import BeautifulSoup
from newspaper import Config
import nltk
from newspaper import Article
import pandas as pd
from selenium import webdriver

# Get the punkt package for sentence breakdown & download at each run to ensure that package is updated
nltk.download('punkt')


class Scraper:
    """
    A web scraper to scrape text information from the web pages
    A web page is a search result from Google with user defined search text
    The final results will be saved into a csv file
    """
    def __init__(self, search_type):
        user_agent = 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:24.0) Gecko/20100101 Firefox/24.0'
        self.config = Config()
        self.config.browser_user_agent = user_agent
        self.config.request_timeout = 120
        self.search_type = search_type
        self.prepare_input()

    def prepare_input(self):
        """
        Get search text from input
        Search_Type (input 1 or 2):
            There are 2 different search types which need to be selected before input the search text:
                Type 1: Search the results using the search text input by user manually
                Type 2: Search the results using a csv file with multiple search text stored in it, the scraper will
                search for the results row by row
        """
        # get search text from user's input
        if self.search_type == 1:
            search_text = input("Enter the Search Text: ")
            if search_text:
                search_text = "+".join(search_text.split())
                articles = self.scraping(search_text)
                cols = ['Link', 'Search_text', 'Text']
                df_article = pd.DataFrame(articles, columns=cols)
                df_article.to_csv('scraper_results.csv', index=False, encoding='utf-8-sig')

            else:
                print('Invalid search text')

        # get search_text from a csv file row by row
        elif self.search_type == 2:
            cols = ['Link', 'Search_text', 'Text']
            df_articles = pd.DataFrame(columns=cols)
            file_path = input("Enter file path location: ")
            search_text_df = pd.read_csv(file_path)
            for index, row in search_text_df.iterrows():
                search_text = row[0]
                if search_text:
                    print("Searching for: {}".format(search_text))
                    search_text = "+".join(search_text.split())
                    article = self.scraping(search_text)
                    df_article = pd.DataFrame(article, columns=cols)
                    df_articles = df_articles.append(df_article, ignore_index=True)
                else:
                    print('Invalid search text')
            df_articles.to_csv('scraper_results.csv', index=False, encoding='utf-8-sig')

        else:
            print("Error: No such search type")

    def scraping(self, search_text):
        """
        Scrape results by search text
        The scraper will get into each valid link listing on Google searching page,
        and scrape the text information on the web page
        """
        articles = []
        # search on Google
        template = 'https://www.google.com/search?q={}'
        url = template.format(search_text)
        # load selenium webdriver
        driver = webdriver.Chrome('chromedriver')
        driver.get(url)
        # extract valid urls in the search page
        soup = BeautifulSoup(driver.page_source, 'lxml')
        cards = soup.find_all('div', attrs={'class': 'g'})
        # iterate each valid url
        for card in cards:
            link = card.find('a', href=True)
            if link:
                # extract text from the web page
                article = self.get_article(link['href'])
                articles.append([link['href'], search_text, article])

        return articles

    def get_article(self, link):
        # Get the context on the web page
        article = Article(link, config=self.config)
        try:
            article.download()
            article.parse()
            article.nlp()
            return article.text
        except:
            pass


if __name__ == '__main__':
    search_type = int(input("Enter the Search Type (1 or 2): "))
    Scraper(search_type)
