import os
import sys
import shutil
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import List
from pytz import timezone
from loguru import logger

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from args import get_args

current_dir = os.path.abspath(os.path.join(parent_dir, "fact_checker/scrapy"))
sys.path.append(current_dir)
from run_spiders import parse_parallel


def create_factcheck_scheduler(
    factcheck_dir: str,
    start_date: str,
    end_date: str,
    start_page: int,
    end_page: int,
    title_keys: str,
    tags: str,
    political_keywords: List[str],
    spiders: List[str],
) -> BackgroundScheduler:

    scheduler = BackgroundScheduler()  # create a scheduler
    scheduler.configure(timezone=timezone("US/Eastern"))
    trigger = CronTrigger(
        hour="6", minute="00"
    )  # create a CronTrigger to run at 6 AM every day
    # trigger = CronTrigger(minute="*")  # create a CronTrigger to run every minute

    scheduler.add_job(
        parse_parallel,
        trigger=trigger,
        args=[
            factcheck_dir,
            start_date,
            end_date,
            start_page,
            end_page,
            title_keys,
            tags,
            political_keywords,
            spiders,
        ],
    )  # add the job to the scheduler

    return scheduler


if __name__ == "__main__":

    # To create necessary directories for model and data if not already present
    args = get_args()
    assets_dir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), "./../../")), args.assets_dir
    )
    data_dir = os.path.join(assets_dir, args.data_dir)
    factcheck_dir = os.path.join(data_dir, args.factcheck_dir)

    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(factcheck_dir, exist_ok=True)

    start_date = args.start_date
    end_date = args.end_date
    start_page = args.start_page
    end_page = args.end_page
    title_keys = args.title_keys
    tags = args.tags
    political_keywords = args.political_keywords
    spiders = args.spiders

    # Create and start fact check scheduler
    logger.info(f"Creating fact check scheduler ...")
    factcheck_scheduler = create_factcheck_scheduler(
        factcheck_dir,
        start_date,
        end_date,
        start_page,
        end_page,
        title_keys,
        tags,
        political_keywords,
        spiders,
    )
    logger.info(f"Starting fact check scheduler ...")
    factcheck_scheduler.start()
    logger.info(f"Fact check scheduler running ...")

    try:  # keep the script running
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):  # shut down the scheduler on exit
        logger.info(f"Shutting down fact check scheduler ...")
        factcheck_scheduler.shutdown()
        logger.info(f"Fact check scheduler shut down.")
