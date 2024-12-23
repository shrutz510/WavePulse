import os
import json
from loguru import logger
from typing import List, Dict, Any, Optional
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging
from ruling_module import create_new_ruling
from political_module import filter_political_articles
from duplicates_module import find_duplicates

# Import spiders here
from factcheck_crawler.spiders.apnews_spider import APNewsSpider
from factcheck_crawler.spiders.checkyourfact_spider import CheckYourFactSpider
from factcheck_crawler.spiders.factcheck_spider import FactCheckSpider
from factcheck_crawler.spiders.leadstories_spider import LeadStoriesSpider
from factcheck_crawler.spiders.politifact_spider import PolitifactSpider
from factcheck_crawler.spiders.snopes_spider import SnopesSpider
from factcheck_crawler.spiders.truthorfiction_spider import TruthOrFictionSpider


def merge_json_files(source_dir: str, output_file: str) -> None:
    """
    Merges JSON files from the source directory into a single JSON file.

    Parameters:
        - source_dir (str): The directory containing the JSON files to be merged.
        - output_file (str): The path to the output file where the merged JSON content will be saved.
    """

    merged_data = []

    for filename in os.listdir(source_dir):
        if filename.endswith("_output.json"):
            source_path = os.path.join(source_dir, filename)
            try:
                with open(source_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    merged_data.extend(data)
            except (Exception, json.JSONDecodeError) as e:
                logger.error(f"Error processing {filename}: {e}")

    try:
        with open(output_file, "w", encoding="utf-8") as output:
            json.dump(merged_data, output, indent=4)
        logger.info(f"Total articles scraped: {len(merged_data)}")
        logger.info(f"Merged scraped articles items into {output_file}")
    except Exception as e:
        logger.error(f"Error writing merged JSON to file {output_file}: {e}")

    return None


def delete_json_files(source_dir: str) -> None:
    """
    Deletes JSON files from the source directory.

    Parameters:
        - source_dir (str): The directory containing the JSON files to be merged.

    """

    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            source_path = os.path.join(source_dir, filename)
            try:
                os.remove(source_path)
            except Exception as e:
                logger.error(f"Error deleting {filename} from {source_path}: {e}")

    return None


def run_spiders_and_merge(
    factcheck_dir: str,
    spider_args: Dict[str, Any],
    political_keywords: List[str],
    spider_list: Optional[List[str]],
) -> None:
    """
    Runs selected spiders and merges the resulting JSON files.

    Parameters:
        - spider_args (Dict[str, Any]): A dictionary of arguments to be passed to the spiders.
        - political_keywords (List[str]): A list of keywords to identify political content.
        - spider_list (Optional[List[str]]): A list of spiders to run. If None, all spiders are run.

    Example:
        >>> spider_args = {'start_date': '01-01-2024', 'end_date': '01-31-2024'}
        >>> political_keywords = ['election', 'policy', 'government']
        >>> spider_list = ['spider1', 'spider2']
        >>> run_spiders_and_merge(spider_args, political_keywords, spider_list)
    """

    configure_logging()
    settings = get_project_settings()
    process = CrawlerProcess(settings)

    spider_classes = {
        "apnews": APNewsSpider,
        "checkyourfact": CheckYourFactSpider,
        "factcheck": FactCheckSpider,
        "leadstories": LeadStoriesSpider,
        "politifact": PolitifactSpider,
        "snopes": SnopesSpider,
        "truthorfiction": TruthOrFictionSpider,
    }

    if spider_list is None:
        spiders_to_run = list(spider_classes.values())
    else:
        spiders_to_run = [
            spider_classes[spider] for spider in spider_list if spider in spider_classes
        ]

    for spider in spiders_to_run:
        try:
            process.crawl(spider, **spider_args)
        except Exception as e:
            logger.error(f"Error starting spider {spider.name}: {e}")

    process.start()

    source_dir = "./"
    output_file = f"./merged_json_files.json"

    merge_json_files(source_dir, output_file)

    with open(output_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    data = create_new_ruling(data)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    delete_json_files(source_dir)
    filter_political_articles(factcheck_dir, data, political_keywords)
    find_duplicates(factcheck_dir)

    return None


def parse_parallel(
    factcheck_dir: str,
    start_date: str,
    end_date: str,
    start_page: int,
    end_page: int,
    title_keys: str,
    tags: str,
    political_keywords: List[str],
    spiders: List[str],
) -> None:

    """
    Main function to parse arguments and run the spiders.
    This script runs the specified spiders with the given parameters to scrape and merge fact-checking articles.

    Methods:
        - parse_parallel(): Parses arguments and runs the spiders with the given parameters.

    Example:
        >>> Run the script with specific parameters
        >>> python run_spiders.py --start_date='01-01-2024' --end_date='06-01-2024' --title_keys='election' --spiders='apnews, politifact'
    """

    spider_args = {
        "start_date": start_date,
        "end_date": end_date,
        "start_page": start_page,
        "end_page": end_page,
        "title_keys": title_keys,
        "tags": tags,
    }

    run_spiders_and_merge(factcheck_dir, spider_args, political_keywords, spiders)

    return None
