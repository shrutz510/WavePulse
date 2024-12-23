import scrapy
import time
import re
from datetime import datetime
from typing import Generator, Dict, Any
from scrapy.http import Response
from scrapy import Request


class SnopesSpider(scrapy.Spider):
    """
    Spider for scraping articles from Snopes.com.

    Parameters:
        - start_date (str, optional): The start date for filtering articles in MM-DD-YYYY format.
        - end_date (str, optional): The end date for filtering articles in MM-DD-YYYY format.
        - title_keys (str, optional): Comma-separated keywords for filtering articles by title.
        - tags (str, optional): Comma-separated tags for filtering articles by tags.
        - start_page (int, optional): The start page number for pagination (default is 1).
        - end_page (int, optional): The end page number for pagination (default is 60).

    Methods:
        - __init__(start_date, end_date, title_keys, tags, start_page, end_page, *args, **kwargs): Initializes the spider with optional date, keyword, and pagination filters.
        - parse(response): Parses the main page for articles and follows links to individual articles.
        - parse_article(response): Parses individual articles for detailed information.
        - closed(reason): Logs the total time taken for scraping when the spider closes.

    Example:
        >>> scrapy crawl snopes -O snopes.json -a start_date='01-01-2024' -a title_keys='election' -a tags='politics'
        >>> spider = SnopesSpider(start_date='01-01-2024', title_keys='election')
        >>> crawler = CrawlerProcess()
        >>> crawler.crawl(spider)
        >>> crawler.start()
    """

    name = "snopes"
    allowed_domains = ["snopes.com"]

    custom_settings = {
        "FEEDS": {
            "snopes_output.json": {
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
        tags: str = None,
        start_page: int = None,
        end_page: int = None,
        *args,
        **kwargs,
    ) -> None:

        super(SnopesSpider, self).__init__(*args, **kwargs)
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
            self.end_page = 60

        # Check if the start page is not after the end page
        if self.start_page > self.end_page:
            raise ValueError(
                f"start_page {self.start_page} cannot be greater than end_page {self.end_page}"
            )

        self.start_urls = [
            f"https://www.snopes.com/fact-check/?pagenum={i}"
            for i in range(self.start_page, self.end_page + 1)
        ]

        # Set the title keywords for filtering
        if title_keys:
            self.title_keys = title_keys.split(",")
        else:
            self.title_keys = []

        # Set the tags for filtering
        if tags:
            self.tags = tags.split(",")
        else:
            self.tags = []

    def parse(self, response: Response) -> Generator[Request, None, None]:
        # Extract fact-checking articles from the page
        articles = response.css("div.article_list_cont div.article_wrapper")
        for article in articles:
            # Extract relevant information from each article
            title_tag = article.css("a.outer_article_link_wrapper")
            title = article.css("h3.article_title::text").get()  # title
            url = title_tag.css("::attr(href)").get()  # url
            author = article.css("span.author_name::text").get()  # url
            date = article.css("span.article_date::text").get()  # date

            # Convert date to the required format
            date = re.search(r".*?\d{4}", date.strip()).group(0)
            date = re.sub(r"\.", "", date)
            for fmt in ("%b %d, %Y", "%B %d, %Y"):
                try:
                    parsed_date = datetime.strptime(date, fmt)
                    date = parsed_date.strftime("%m-%d-%Y")
                    break
                except ValueError:
                    continue
            else:
                self.logger.error(f"Error parsing date: {date}")
                continue
            start_date = self.start_date.strftime("%m-%d-%Y")
            end_date = self.end_date.strftime("%m-%d-%Y")

            # Filter by date range
            if date < start_date or date > end_date:
                continue

            # Filter articles by title keywords
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
                    url,
                    self.parse_article,
                    meta={
                        "title": title.strip() if title else None,
                        "url": response.urljoin(url),
                        "date": date,
                        "author": author.strip() if author else None,
                    },
                )

    def parse_article(
        self, response: Response
    ) -> Generator[Dict[str, Any], None, None]:
        # Get the passed metadata
        title = response.meta.get("title")
        date = response.meta.get("date")
        author = response.meta.get("author")

        claim_content = response.css("div.claim_cont::text").get()  # content
        context_content = response.css(
            "p.fact_check_info_description::text"
        ).get()  # content
        if claim_content and context_content:
            content = f"{claim_content} {context_content}"
        else:
            content = claim_content

        tags = response.css("div.tag_wrapper a::text").getall()  # tags
        tags = [tag.strip() for tag in tags]  # convert tags to list

        # Filter articles by tags
        if self.tags:
            if not any(tag.lower() in [t.lower() for t in tags] for tag in self.tags):
                return

        content = content.replace("\t", "").replace("\n", "")  # format content

        ruling = response.css("div.rating_title_wrap::text").get()  # ruling

        # Yield the complete data
        yield {
            "title": title if title else None,
            "url": response.url,
            "date": date if date else None,
            "author": author if author else None,
            "content": content.strip() if content else None,
            "tags": tags if tags else None,
            "ruling": ruling.strip() if ruling else None,
            "website": "snopes",
        }

    def closed(self, reason) -> None:
        end_time = time.time()  # Record the end time
        duration = end_time - self.start_time
        self.logger.info(f"Scraping finished in {duration:.2f} seconds")
