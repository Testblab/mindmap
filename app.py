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



import streamlit as st
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import json

st.set_page_config(page_title="Mind Map des Produits", layout="wide")
st.title("🧠 Générateur de Mind Map de Produits d'une Entreprise")

# Entrées utilisateur
company = st.text_input("Nom de l'entreprise")
year = st.text_input("Année (ex : 2024)")

if st.button("Générer") and company and year:
    with st.spinner("Recherche des produits et fonctionnalités..."):

        def get_links(query):
            try:
                return list(search(query, num_results=10))
            except Exception as e:
                return []

        def extract_info_from_url(url):
            try:
                resp = requests.get(url, timeout=5)
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = soup.get_text()
                return text
            except:
                return ""

        def extract_mindmap(company, year):
            query = f"{company} produits fonctionnalités {year}"
            links = get_links(query)
            data = {}
            for link in links:
                content = extract_info_from_url(link)
                if company.lower() in content.lower():
                    # Extraction très basique par phrases contenant "produit" ou "fonctionnalité"
                    lines = content.split(".\n")
                    for line in lines:
                        if "produit" in line.lower() or "fonctionnalité" in line.lower():
                            for word in line.split():
                                if word.istitle():
                                    prod = word.strip(".,:;()[]")
                                    if prod not in data:
                                        data[prod] = []
                                    if "fonctionnalité" in line.lower():
                                        data[prod].append(line.strip())
            return data

        mindmap_data = extract_mindmap(company, year)

        if not mindmap_data:
            st.warning("Aucune information trouvée.")
        else:
            st.success("Données récupérées !")

            import streamlit.components.v1 as components

            # Générer le HTML jsMind
            def make_jsmind_json(data):
                result = [{"id": "root", "isroot": True, "topic": company}]
                pid = 0
                for i, prod in enumerate(data):
                    prod_id = f"p{i}"
                    result.append({"id": prod_id, "parentid": "root", "topic": prod})
                    for j, feat in enumerate(data[prod]):
                        result.append({"id": f"f{i}{j}", "parentid": prod_id, "topic": feat[:50] + ('...' if len(feat) > 50 else '')})
                return result

            tree_data = make_jsmind_json(mindmap_data)
            tree_json = json.dumps(tree_data)

            html_code = f"""
            <div id="jsmind_container" style="width:100%;height:500px;border:1px solid #ccc"></div>
            <script src="https://cdn.jsdelivr.net/npm/jsmind@0.4.6/es6/jsmind.js"></script>
            <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/jsmind@0.4.6/style/jsmind.css" />
            <script>
                const mind = {{"meta":{{"name":"mindmap"}},"format":"node_tree","data":{json.dumps(tree_data[0])}}};
                mind.data.children = {json.dumps(tree_data[1:])};
                const options = {{
                    container: 'jsmind_container',
                    editable: false,
                    theme: 'primary'
                }};
                const jm = new jsMind(options);
                jm.show(mind);
            </script>
            """
            components.html(html_code, height=550)

else:
    st.info("Veuillez entrer un nom d'entreprise et une année pour générer la carte.")
