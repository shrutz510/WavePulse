import json
import os
from loguru import logger


def is_political(article: dict, political_keywords: list) -> bool:
    """
    Determines if an article is political based on the presence of specific keywords.

    Parameters:
        - article (dict): A dictionary representing the article with keys 'title', 'content', and 'tags'.
        - political_keywords (list): A list of keywords to identify political content.

    Returns:
        - bool: True if the article is political, False otherwise.

    Example:
        >>> article = {
        >>>     'title': 'Elections 2024: Major Updates',
        >>>     'content': 'The upcoming elections are drawing a lot of attention...',
        >>>     'tags': ['election', 'politics', 'vote']
        >>> }
        >>> political_keywords = ['election', 'policy', 'government']
        >>> is_political(article, political_keywords)
        True
    """

    title = article.get("title", "").lower()
    content = article.get("content", "").lower() if article.get("content") else ""
    tags = article.get("tags", [])

    for keyword in political_keywords:
        if keyword in title or keyword in content:
            return True
        if tags:
            for tag in tags:
                if tag and keyword in tag.lower():
                    return True

    return False


def filter_political_articles(
    factcheck_dir: str, new_articles: list, political_keywords: list
) -> None:
    """
    Filters political articles from a JSON file and adds new political articles to the existing file.

    Parameters:
        - factcheck_dir (str): The directory with the political_articles.json file.
        - new_articles (list): A list of new articles to be filtered for political content.
        - political_keywords (list): A list of keywords to identify political articles.

    Example:
        >>> new_articles = [{'title': 'Elections 2024', 'content': '...'}, {'title': 'Sports update', 'content': '...'}]
        >>> political_keywords = ['election', 'policy', 'government']
        >>> filter_political_articles(new_articles, political_keywords)
    """

    file_path = f"{factcheck_dir}/political_articles.json"

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            articles = json.load(file)
    else:
        articles = []

    political_articles = [
        article for article in articles if is_political(article, political_keywords)
    ]
    previous_articles = len(political_articles)
    political_articles.extend(
        [
            article
            for article in new_articles
            if is_political(article, political_keywords)
        ]
    )

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(political_articles, file, indent=4)

    logger.info(
        f"Political articles scraped: {len(political_articles) - previous_articles}"
    )
    logger.info(f"Total political articles: {len(political_articles)}")
    logger.info(f"Political articles added to: {file_path}")

    return None
