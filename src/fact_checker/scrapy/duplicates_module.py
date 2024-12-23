import json
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# Convert date
def parse_date(date_str):
    return datetime.strptime(date_str, "%m-%d-%Y")


# Check if two dates are within the same week
def is_within_same_week(date1, date2):
    return abs((date1 - date2).days) < 7


def find_duplicates(factcheck_dir):

    input_file_path = f"{factcheck_dir}/political_articles.json"
    output_file_path = f"{factcheck_dir}/deduplicated_articles.json"

    with open(input_file_path, "r") as file:
        articles = json.load(file)

    articles_filtered = [article for article in articles if article["content"]]
    texts = [article["content"] for article in articles_filtered]
    dates = [parse_date(article["date"]) for article in articles_filtered]
    vectorizer = TfidfVectorizer().fit_transform(texts)
    cosine_sim_matrix = 1 - pairwise_distances(vectorizer, metric="cosine")

    clusters = defaultdict(list)
    duplicate_indices = set()
    cluster_id = 0

    for i in tqdm(range(len(articles_filtered)), desc="Finding duplicates"):
        if i in duplicate_indices:
            continue
        current_cluster = [i]
        for j in range(i + 1, len(articles_filtered)):
            if (
                cosine_sim_matrix[i][j] > 0.4
                and articles_filtered[i]["website"] != articles_filtered[j]["website"]
                and is_within_same_week(dates[i], dates[j])
            ):
                current_cluster.append(j)
                duplicate_indices.add(j)
        if len(current_cluster) > 1:
            for index in current_cluster:
                clusters[cluster_id].append(articles_filtered[index])
            cluster_id += 1

    duplicates = []
    for cluster_id, cluster_articles in clusters.items():
        duplicates.append(
            {
                "cluster_id": cluster_id,
                "articles": [
                    {
                        "title": article["title"],
                        "url": article["url"],
                        "author": article["author"],
                        "content": article["content"],
                        "date": article["date"],
                        "website": article["website"],
                        "ruling": article["ruling-unified"],
                    }
                    for article in cluster_articles
                ],
            }
        )

    unique_articles_with_content = [
        article
        for idx, article in enumerate(articles_filtered)
        if idx not in duplicate_indices
    ]
    unique_articles_no_content = [
        article for article in articles if not article["content"]
    ]
    unique_articles = unique_articles_with_content + unique_articles_no_content

    logger.info(f"Total unique articles: {len(unique_articles)}")
    logger.info(f"Duplicate clusters: {len(duplicates)}")

    all_articles = unique_articles + duplicates
    sorted_articles = sorted(
        all_articles,
        key=lambda x: parse_date(x["date"])
        if "date" in x
        else parse_date(x["articles"][0]["date"]),
        reverse=True,
    )

    with open(output_file_path, "w") as file:
        json.dump(sorted_articles, file, indent=4)

    logger.info(f"All {len(sorted_articles)} articles saved to {output_file_path}")

    return None
