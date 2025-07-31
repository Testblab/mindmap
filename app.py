import streamlit as st
import requests
from bs4 import BeautifulSoup
import graphviz

def scrape_products_and_features(company: str, year: str):
    """Scrape web search results for product information.

    The function queries DuckDuckGo for public information about the
    company's products in the given year. It then extracts naive
    product/feature pairs from the search result snippets.
    """
    query = f"{company} {year} products"
    url = "https://duckduckgo.com/html/"
    try:
        resp = requests.get(url, params={"q": query}, timeout=10)
    except requests.RequestException:
        return {}
    if resp.status_code != 200:
        return {}
    soup = BeautifulSoup(resp.text, "html.parser")
    data = {}
    for result in soup.select('.result'):
        snippet_tag = result.select_one('.result__snippet')
        if not snippet_tag:
            continue
        snippet = snippet_tag.get_text(" ").strip()
        if not snippet:
            continue
        words = [w.strip('.,;:!?()[]') for w in snippet.split() if w.isalpha()]
        if not words:
            continue
        product = words[0]
        features = words[1:5]
        if product:
            data.setdefault(product, []).extend(features)
    return data

def create_mindmap(data: dict, company: str):
    """Create a mind map using Graphviz and display it in Streamlit."""
    graph = graphviz.Digraph(format='png')
    graph.node(company, shape='ellipse')
    for product, features in data.items():
        graph.node(product, shape='ellipse')
        graph.edge(company, product)
        for feature in features:
            node_id = f"{product}_{feature}"
            graph.node(node_id, label=feature, shape='box')
            graph.edge(product, node_id)
    st.graphviz_chart(graph)


def main():
    st.title("Mindmap des produits")
    company = st.text_input("Nom de l'entreprise")
    year = st.text_input("Année")
    if st.button("Générer"):
        if not company or not year:
            st.error("Veuillez renseigner tous les champs.")
        else:
            with st.spinner("Scraping des données..."):
                data = scrape_products_and_features(company, year)
            if not data:
                st.warning("Aucun produit trouvé.")
            else:
                create_mindmap(data, company)

if __name__ == "__main__":
    main()
