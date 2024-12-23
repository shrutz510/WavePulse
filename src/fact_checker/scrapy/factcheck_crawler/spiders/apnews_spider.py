import scrapy
import time
import re
from datetime import datetime
from typing import Generator, Dict, Any
from scrapy.http import Response
from scrapy import Request


class APNewsSpider(scrapy.Spider):
    """
    Spider for scraping articles from AP News Fact-Checking hub.

    Parameters:
        - start_date (str, optional): The start date for filtering articles in MM-DD-YYYY format.
        - end_date (str, optional): The end date for filtering articles in MM-DD-YYYY format.
        - title_keys (str, optional): Comma-separated keywords for filtering articles by title.
        - start_page (int, optional): The start page number for pagination (default is 1).
        - end_page (int, optional): The end page number for pagination (default is 5).

    Methods:
        - __init__(start_date, end_date, title_keys, start_page, end_page, *args, **kwargs): Initializes the spider with optional date, keyword, and pagination filters.
        - parse(response): Parses the main page for articles and follows links to individual articles.
        - parse_article(response): Parses individual articles for detailed information.
        - closed(reason): Logs the total time taken for scraping when the spider closes.

    Example:
        >>> scrapy crawl apnews -O apnews.json -a start_date='01-01-2024' -a title_keys='election'
        >>> spider = APNewsSpider(start_date='01-01-2024', title_keys='election')
        >>> crawler = CrawlerProcess()
        >>> crawler.crawl(spider)
        >>> crawler.start()
    """

    name = "apnews"
    allowed_domains = ["apnews.com"]

    custom_settings = {
        "FEEDS": {
            "apnews_output.json": {
                "format": "json",
                "encoding": "utf8",
                "store_empty": False,
                "overwrite": True,
                "indent": 4,
            }
        }
    }

    def __init__(
        self,
        start_date: str = None,
        end_date: str = None,
        title_keys: str = None,
        start_page: int = None,
        end_page: int = None,
        *args,
        **kwargs,
    ) -> None:

        super(APNewsSpider, self).__init__(*args, **kwargs)
        self.start_time = time.time()  # Record the start time

        # Validate and set the start date
        if start_date:
            try:
                self.start_date = datetime.strptime(start_date, "%m-%d-%Y").date()
            except ValueError:
                raise ValueError(
                    f"Invalid start_date format: {start_date}. Expected format: MM-DD-YYYY"
                )
        else:
            self.start_date = datetime.now().date()

        # Validate and set the end date
        if end_date:
            try:
                self.end_date = datetime.strptime(end_date, "%m-%d-%Y").date()
            except ValueError:
                raise ValueError(
                    f"Invalid end_date format: {end_date}. Expected format: MM-DD-YYYY"
                )
        else:
            self.end_date = datetime.now().date()

        # Check if the start date is not later than the end date
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date {self.start_date} cannot be later than end_date {self.end_date}"
            )

        # Validate and set the start page for pagination
        if start_page:
            try:
                self.start_page = int(start_page)
            except ValueError:
                raise ValueError(
                    f"Invalid start_page format: {start_page}. Expected format: int"
                )
        else:
            self.start_page = 1

        # Validate and set the end page for pagination
        if end_page:
            try:
                self.end_page = int(end_page)
            except ValueError:
                raise ValueError(
                    f"Invalid end_page format: {end_page}. Expected format: int"
                )
        else:
            self.end_page = 15

        # Check if the start page is not after the end page
        if self.start_page > self.end_page:
            raise ValueError(
                f"start_page {self.start_page} cannot be greater than end_page {self.end_page}"
            )

        self.start_urls = [
            f"https://apnews.com/hub/fact-checking?p={i}"
            for i in range(self.start_page, self.end_page + 1)
        ]

        # Set the title keywords for filtering
        if title_keys:
            self.title_keys = title_keys.split(",")
        else:
            self.title_keys = []

    def parse(self, response: Response) -> Generator[Request, None, None]:
        # Extract the links to the articles
        articles = response.css("div.PageList-items-item")
        for article in articles:
            title = article.css("span.PagePromoContentIcons-text::text").get()  # title
            url = article.css("h3.PagePromo-title a::attr(href)").get()  # url

            # Filter by title keyword
            if self.title_keys:
                if not any(
                    keyword
                    for keyword in self.title_keys
                    if keyword.lower() in title.lower()
                ):
                    return

            # Follow url to article page
            if url:
                if "apnews.com/ap-fact-check" not in url:
                    yield response.follow(
                        response.urljoin(url),
                        self.parse_article,
                        meta={"title": title.strip()},
                    )

    def parse_article(
        self, response: Response
    ) -> Generator[Dict[str, Any], None, None]:
        # Get the passed metadata
        title = response.meta["title"]

        # Get the date & time
        timestamp = response.css("bsp-timestamp::attr(data-timestamp)").get()
        if timestamp:
            struct_time = time.localtime(int(timestamp) / 1000)
            date = datetime.strptime(
                time.strftime("%Y-%m-%d", struct_time), "%Y-%m-%d"
            ).date()  # date
            time_of_day = time.strftime("%H:%M:%S", struct_time)  # time

            author = response.css(
                "div.Page-authors a.Link::text, div.Page-authors span::text"
            ).get()
            if author == None:
                author = response.css("div.Page-authors::text").get()  # author
                if author:
                    author = re.sub(
                        r"^By(?:\s|\xa0)+", "", author
                    ).strip()  # format author
                    author = re.sub(
                        r"^By(?:\s|\xa0|\n)+", "", author
                    ).strip()  # format author

            claim_content = (
                response.css('div.RichTextStoryBody p:contains("CLAIM:")')
                .xpath("string(.)")
                .get()
            )  # content
            assessment_content = (
                response.css('div.RichTextStoryBody p:contains("AP’S ASSESSMENT:")')
                .xpath("string(.)")
                .get()
            )  # content
            facts_content = (
                response.css('div.RichTextStoryBody p:contains("THE FACTS:")')
                .xpath("string(.)")
                .get()
            )  # content
            content = f"{claim_content} {assessment_content} {facts_content}"

            # Filter by date range
            if date < self.start_date or date > self.end_date:
                return

            # Response
            yield {
                "title": title if title else None,
                "url": response.url,
                "date": date.strftime("%m-%d-%Y") if date else None,
                "time": time_of_day if time_of_day else None,
                "author": author if author else None,
                "content": content.strip() if content else None,
                "ruling": "only-analysis-no-label",
                "website": "apnews",
            }

    def closed(self, reason: str) -> None:
        end_time = time.time()  # Record the end time
        duration = end_time - self.start_time
        self.logger.info(f"Scraping finished in {duration:.2f} seconds")
