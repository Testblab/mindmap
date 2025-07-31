"""
MindMap Application
===================

This module implements a simple Flask web application for generating interactive
mind maps of a company's products and their associated features for a given year.
The front‑end provides a form where a user can enter the name of a company
and a year. When the user clicks the "Générer" button, the back‑end scrapes
publicly available web pages and social media posts relevant to that company
for the specified year. It then parses the text to extract product names and
descriptions, organising them into a hierarchical structure that the
front‑end can visualise as a mind map.

To keep the example self‑contained and understandable, the scraping logic
implemented here is deliberately simple. It performs a Google search using
the `googlesearch` library and fetches the HTML from the top few results.
From these pages it looks for instances of words like "produit", "service"
or "produits" followed by capitalised words as a naive proxy for product
names. It then collects sentences containing those names as potential
features. In real applications you would want to rely on dedicated APIs,
structured datasets or more advanced natural language processing to improve
accuracy and respect the terms of service of the sites you scrape.

Usage:
    1. Install the required dependencies: `pip install flask bs4 googlesearch-python requests`
    2. Start the server by running this module: `python app.py`
    3. Visit http://localhost:5000/ in your browser to use the app.

This file should be executed from the repository root or with the working
directory set to the same folder containing this file.
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Iterable, Optional

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # type: ignore
import requests
from bs4 import BeautifulSoup  # type: ignore
try:
    from googlesearch import search  # type: ignore
except ImportError:
    # Provide a clear error message if googlesearch isn't installed.
    raise RuntimeError(
        "The 'googlesearch' package is required. Please install it with: "
        "pip install googlesearch-python"
    )


app: Flask = Flask(__name__)
CORS(app)  # enable CORS so that fetch requests from the same origin work without issues


def scrape_company_products(company_name: str, year: str, *, max_results: int = 5) -> List[Dict[str, Any]]:
    """Scrape search results for the given company and year and extract products and features.

    The function performs a Google search for the company name and year along
    with French keywords for "product" and "features". It then iterates over
    the top ``max_results`` search results, downloads the page content, and
    uses regular expressions to find potential product names and feature
    descriptions.

    Args:
        company_name: The name of the company to search for.
        year: The year to include in the search query.
        max_results: The maximum number of search results to parse.

    Returns:
        A list of dictionaries. Each dictionary has a 'name' key for the
        product name and a 'features' key which is a list of strings
        describing the product's features.

    Note:
        This is a very naive implementation intended for demonstration
        purposes only. It does not respect robots.txt nor the terms of
        service of the websites it scrapes. For production use, please
        consider using official APIs or datasets and always comply with
        websites' usage policies.
    """

    query: str = f"{company_name} {year} produit fonctionnalités"
    product_features: Dict[str, List[str]] = {}

    # Perform a Google search. Using 'lang=fr' prioritises French results.
    results: Iterable[str] = search(query, num_results=max_results, lang="fr")

    for url in results:
        try:
            response = requests.get(url, timeout=10)
            # Skip non‑HTML content types
            content_type: Optional[str] = response.headers.get("Content-Type")
            if content_type and "text/html" not in content_type:
                continue
            html: str = response.text
        except Exception as exc:
            # Skip URLs that can't be fetched
            print(f"[scrape_company_products] Failed to fetch {url}: {exc}")
            continue

        soup: BeautifulSoup = BeautifulSoup(html, "html.parser")
        # Extract visible text from the page
        text: str = soup.get_text(separator=" ")

        # Regex to find product names that follow terms like "produit" or "service"
        # It captures one or more capitalised words (including hyphens) after these keywords.
        product_pattern = re.compile(r"(?:produit|produits|service|services)\s+([A-ZÉÈÂÔÊÀÇ][\w-]*(?:\s+[A-ZÉÈÂÔÊÀÇ][\w-]*)*)", re.I)
        # Find all potential product names
        matches: List[str] = product_pattern.findall(text)
        for match in matches:
            product: str = match.strip()
            # Find sentences containing the product name (within ~200 characters)
            sentence_pattern = re.compile(rf"(\b{re.escape(product)}[^\.{{}}]{{0,200}}\.)", re.IGNORECASE)
            sentences: List[str] = sentence_pattern.findall(text)
            features: List[str] = []
            for sent in sentences:
                cleaned: str = " ".join(sent.split())  # normalise whitespace
                # Only consider sentences that are not too long and contain verbs like "offre", "permet", "comprend"
                if len(cleaned.split()) <= 60 and re.search(r"offre|permet|comprend|inclut|doté", cleaned, re.IGNORECASE):
                    features.append(cleaned)
            if not features:
                # If we didn't find sentences with verbs, still store at least the product name itself
                features.append("Informations limitées sur les fonctionnalités disponibles.")
            if product in product_features:
                # Extend and deduplicate later
                product_features[product].extend(features)
            else:
                product_features[product] = features

    # Construct the result list, deduplicating features
    results_list: List[Dict[str, Any]] = []
    for product, feats in product_features.items():
        unique_features: List[str] = list(dict.fromkeys(feats))
        results_list.append({"name": product, "features": unique_features})

    return results_list


@app.route("/")
def index() -> str:
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate() -> Any:
    """API endpoint to generate mind map data for a given company and year.

    Expects a JSON payload with 'company' and 'year'. Returns a JSON object
    structured according to the format expected by the jsMind library.
    """
    data: Dict[str, Any] = request.get_json(force=True) or {}
    company: str = data.get("company", "").strip()
    year: str = str(data.get("year", "")).strip()
    if not company or not year:
        return jsonify({"error": "Les champs 'company' et 'year' sont obligatoires."}), 400

    # Scrape the products and features
    nodes: List[Dict[str, Any]] = scrape_company_products(company, year)

    # Build the jsMind tree structure
    tree: Dict[str, Any] = {
        "meta": {
            "name": f"{company} {year} produits",
            "author": "MindMap App",
            "version": "1.0"
        },
        "format": "node_tree",
        "data": {
            "id": "root",
            "topic": f"{company} {year}",
            "children": []  # type: ignore
        }
    }

    for idx, item in enumerate(nodes, start=1):
        product_node: Dict[str, Any] = {
            "id": f"product{idx}",
            "topic": item['name'],
            "children": []
        }
        for jdx, feat in enumerate(item['features'], start=1):
            feature_node: Dict[str, Any] = {
                "id": f"product{idx}_feat{jdx}",
                "topic": feat
            }
            product_node["children"].append(feature_node)
        tree["data"]["children"].append(product_node)

    return jsonify(tree)


if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True)