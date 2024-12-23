import scrapy
import time
from datetime import datetime
from dateutil import parser
from typing import Generator, Dict, Any
from scrapy.http import Response
from scrapy import Request


class CheckYourFactSpider(scrapy.Spider):
    """
    Spider for scraping articles from Check Your Fact website.

    Parameters:
        - start_date (str, optional): The start date for filtering articles in MM-DD-YYYY format.
        - end_date (str, optional): The end date for filtering articles in MM-DD-YYYY format.
        - title_keys (str, optional): Comma-separated keywords for filtering articles by title.
        - start_page (int, optional): The start page number for pagination (default is 1).
        - end_page (int, optional): The end page number for pagination (default is 50).

    Methods:
        - __init__(start_date, end_date, title_keys, start_page, end_page, *args, **kwargs): Initializes the spider with optional date, keyword, and pagination filters.
        - parse(response): Parses the main page for articles and follows links to individual articles.
        - parse_article(response): Parses individual articles for detailed information.
        - closed(reason): Logs the total time taken for scraping when the spider closes.

    Example:
        >>> scrapy crawl checkyourfact -O checkyourfact.json -a start_date='01-01-2024' -a title_keys='election'
        >>> spider = CheckYourFactSpider(start_date='01-01-2024', title_keys='election')
        >>> crawler = CrawlerProcess()
        >>> crawler.crawl(spider)
        >>> crawler.start()
    """

    name = "checkyourfact"
    allowed_domains = ["checkyourfact.com"]

    custom_settings = {
        "FEEDS": {
            "checkyourfack_output.json": {
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

        super(CheckYourFactSpider, self).__init__(*args, **kwargs)
        self.start_time = time.time()  # Record the start time

        # Validate and set the start date
        if start_date:
            try:
                self.start_date = datetime.strptime(
                    start_date.strip(), "%m-%d-%Y"
                ).date()
            except ValueError:
                raise ValueError(
                    f"Invalid start_date format: {start_date}. Expected format: MM-DD-YYYY"
                )
        else:
            self.start_date = datetime.now().date()

        # Validate and set the end date
        if end_date:
            try:
                self.end_date = datetime.strptime(end_date.strip(), "%m-%d-%Y").date()
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
            self.end_page = 50

        # Check if the start page is not after the end page
        if self.start_page > self.end_page:
            raise ValueError(
                f"start_page {self.start_page} cannot be greater than end_page {self.end_page}"
            )

        self.start_urls = [
            f"https://checkyourfact.com/page/{i}"
            for i in range(self.start_page, self.end_page + 1)
        ]

        # Set the title keywords for filtering
        if title_keys:
            self.title_keys = title_keys.split(",")
        else:
            self.title_keys = []

    def parse(self, response: Response) -> Generator[Request, None, None]:
        # Extract the links to the articles
        articles = response.css("a[href] > article")
        for article in articles:
            title = article.css("name::text").get()  # title
            if title is None:
                continue
            if "FACT CHECK: " in title:
                title = title.replace("FACT CHECK: ", "")
            else:
                continue

            url = article.xpath("parent::a/@href").get()  # url

            # Filter by title keyword
            if self.title_keys:
                if not any(
                    keyword
                    for keyword in self.title_keys
                    if keyword.lower() in title.lower()
                ):
                    continue

            # Follow url to article page
            if url:
                yield response.follow(
                    url, self.parse_article, meta={"title": title.strip()}
                )

    def parse_article(
        self, response: Response
    ) -> Generator[Dict[str, Any], None, None]:
        # Get the passed metadata
        title = response.meta["title"]

        date_time_str = response.css("time::text").get()
        if date_time_str:
            parsed_datetime = parser.parse(date_time_str)
            date = parsed_datetime.date()  # date
            time = parsed_datetime.time()  # time
        else:
            date = None
            time = None

        author = response.css("author::text").get()  # author
        if " | Fact Check Reporter" in author:
            author = author.replace(" | Fact Check Reporter", "")  # format author

        ruling = response.css("span strong::text").get()  # ruling
        if ruling and "Verdict: " in ruling:
            ruling = ruling.replace("Verdict: ", "").strip()  # format verdict

        allowed_rulings = [
            "Misleading",
            "False",
            "Unsubstantiated",
            "True",
        ]  # rulings must match given list
        if ruling not in allowed_rulings:
            ruling = "only-analysis-no-label"

        content = response.css("p::text, a::text").getall()
        content = " ".join(content) if content else None

        start_idx = content.find("The claim is ")
        if start_idx != -1:
            content = content[start_idx:]

        content = content.replace("\t", "").replace("\n", "").replace("\r", "").strip()

        # Filter by date range
        if date < self.start_date or date > self.end_date:
            return

        yield {
            "title": title.strip() if title else None,
            "url": response.url,
            "date": date.strftime("%m-%d-%Y") if date else None,
            "time": time if time else None,
            "author": author.strip() if author else None,
            "content": content if content else None,
            "ruling": ruling.strip(),
            "website": "checkyourfact",
        }

    def closed(self, reason: str) -> None:
        end_time = time.time()  # Record the end time
        duration = end_time - self.start_time
        self.logger.info(f"Scraping finished in {duration:.2f} seconds")
