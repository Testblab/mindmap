"""
Application Streamlit pour générer une mind‑map des produits et de leurs fonctionnalités
d'une entreprise donnée à une année donnée.

Cette application propose deux champs d'entrée :
  • ``Nom de l'entreprise`` pour spécifier le nom de l'organisation à analyser;
  • ``Année`` pour limiter la recherche à une période précise.

Après avoir renseigné ces champs et cliqué sur le bouton ``Générer``, le backend
effectue les opérations suivantes :

  1. Exécute une requête de recherche sur DuckDuckGo pour récupérer des
     articles et pages mentionnant les produits et fonctionnalités de
     l'entreprise ciblée. La recherche est limitée à un nombre maximum de
     résultats (configurable).
  2. Télécharge le contenu des pages retournées, puis analyse le HTML avec
     BeautifulSoup afin d'extraire des titres de sections (qui servent de
     supposés noms de produits) et des listes (éléments ``<li>``) représentant
     des fonctionnalités associées.
  3. Agrège l'ensemble des produits et fonctionnalités trouvées dans un
     dictionnaire de la forme ``{Produit : [fonctionnalité1, fonctionnalité2,…]}``.
  4. Construit un graphe interactif à l'aide de ``pyvis`` où le nœud central
     représente l'entreprise, les nœuds de premier niveau représentent les
     produits et les nœuds de second niveau les fonctionnalités. Les liens
     permettent de visualiser la relation hiérarchique entre ces éléments.

Le graphe est rendu dans l'interface Streamlit grâce au composant HTML et au
fichier HTML généré par pyvis. Ce code nécessite l'installation préalable des
packages suivants : ``streamlit``, ``beautifulsoup4``, ``requests``,
``duckduckgo_search``, ``networkx`` et ``pyvis``.
"""

import urllib.parse
from typing import Dict, List

import streamlit as st
from bs4 import BeautifulSoup
import requests
import networkx as nx  # type: ignore
from pyvis.network import Network  # type: ignore
import streamlit.components.v1 as components
import json

try:
    # ``duckduckgo_search`` permet de faire des recherches sans clé API et sans
    # dépendance à un moteur comme Google ou Bing. Si ce module n'est pas
    # installé, l'utilisateur devra l'ajouter (``pip install duckduckgo_search``).
    from duckduckgo_search import DDGS  # type: ignore
except ImportError:
    DDGS = None  # type: ignore


def search_company_products(company: str, year: str, max_results: int = 5) -> List[str]:
    """Rechercher des pages web mentionnant des produits et fonctionnalités.

    Cette fonction utilise DuckDuckGo via le package ``duckduckgo_search`` pour
    exécuter une requête textuelle et récupérer les URLs des résultats. Elle
    concatène le nom de l'entreprise, des mots clés sur les produits et
    l'année sélectionnée.

    Args:
        company: Nom de l'entreprise à rechercher.
        year: Année pour limiter la recherche.
        max_results: Nombre maximum d'URLs à retourner.

    Returns:
        Liste d'URLs correspondant aux résultats de recherche.
    """
    query = f"{company} produits fonctionnalités {year}"
    urls: List[str] = []
    # Vérifier si le module ``duckduckgo_search`` est disponible. Dans le cas
    # contraire, lever une exception explicite afin que l'appelant puisse
    # informer l'utilisateur de la nécessité d'installer la dépendance.
    if DDGS is None:
        raise ImportError(
            "Le module duckduckgo_search n'est pas installé. "
            "Veuillez exécuter `pip install duckduckgo_search` pour permettre la recherche."
        )
    # Utilisation du contexte manager pour s'assurer de la fermeture des sessions.
    with DDGS() as ddgs:
        try:
            for result in ddgs.text(query, region="fr-fr", safesearch="Moderate", max_results=max_results):
                href = result.get("href")
                if href:
                    urls.append(href)
        except Exception:
            # En cas d'erreur lors de la recherche, retourner une liste vide sans planter.
            return []
    return urls


def extract_products_features_from_url(url: str, company: str) -> Dict[str, List[str]]:
    """Extraire des produits et leurs fonctionnalités depuis une page web.

    L'approche adoptée ici est heuristique :
    - Les balises de titre (``<h1>``, ``<h2>``, ``<h3>``) sont considérées comme des
      candidats pour des noms de produits.
    - Les listes (``<ul>``, ``<ol>``) situées après un titre sont interprétées
      comme des listes de fonctionnalités associées au produit identifié par le
      dernier titre rencontré.

    Args:
        url: URL de la page à analyser.
        company: Nom de l'entreprise (permet d'ignorer les titres correspondant
                 à l'entreprise elle‑même).

    Returns:
        Un dictionnaire où chaque clé est le nom d'un produit et la valeur une
        liste de fonctionnalités extraites.
    """
    product_map: Dict[str, List[str]] = {}
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return product_map
        soup = BeautifulSoup(response.content, "html.parser")
        # Rechercher les titres pour identifier les produits
        for heading in soup.find_all(["h1", "h2", "h3"]):
            product_name = heading.get_text(separator=" ").strip()
            # Ignorer les titres vides ou extrêmement longs (plus de 15 mots)
            if not product_name or len(product_name.split()) > 15:
                continue
            # Initialiser une liste vide pour ce produit si elle n'existe pas déjà
            product_map.setdefault(product_name, [])
        # Extraire les fonctionnalités depuis les listes de la page
        for ul in soup.find_all(["ul", "ol"]):
            items = [li.get_text(separator=" ").strip() for li in ul.find_all("li")]
            if not items:
                continue
            # Trouver le titre le plus proche précédant cette liste pour l'associer
            parent_heading = ul.find_previous(["h1", "h2", "h3"])
            if not parent_heading:
                # S'il n'y a pas de titre précédent, associer les fonctionnalités à
                # une catégorie générique.
                parent_name = "Fonctionnalités diverses"
            else:
                parent_name = parent_heading.get_text(separator=" ").strip() or "Fonctionnalités diverses"
            product_map.setdefault(parent_name, [])
            for item in items:
                if item and item not in product_map[parent_name]:
                    product_map[parent_name].append(item)
        return product_map
    except Exception:
        # En cas d'erreur réseau ou de parsing, retourner un dictionnaire vide.
        return {}


def scrape_company_products(company: str, year: str, max_results: int = 5) -> Dict[str, List[str]]:
    """Aggreguer les produits et fonctionnalités via recherche et scraping.

    Combine ``search_company_products`` et ``extract_products_features_from_url`` pour
    constituer un dictionnaire exhaustif des produits et de leurs fonctionnalités.

    Args:
        company: Nom de l'entreprise à analyser.
        year: Année ciblée pour la recherche.
        max_results: Nombre maximum de pages à analyser.

    Returns:
        Dictionnaire ``{Produit : [fonctionnalité1, fonctionnalité2, …]}``.
    """
    aggregated: Dict[str, List[str]] = {}
    urls = search_company_products(company, year, max_results=max_results)
    for url in urls:
        product_map = extract_products_features_from_url(url, company)
        for product, features in product_map.items():
            aggregated.setdefault(product, [])
            for feature in features:
                if feature not in aggregated[product]:
                    aggregated[product].append(feature)
    return aggregated


def build_mind_map(company: str, data: Dict[str, List[str]]) -> Network:
    """Créer un graphe pyvis représentant la mind‑map des produits et fonctionnalités.

    Le nœud racine correspond à l'entreprise. Les produits sont des nœuds
    connectés au nœud racine, et chaque fonctionnalité est reliée à son
    produit.

    Args:
        company: Nom de l'entreprise (utilisé comme nœud central).
        data: Dictionnaire ``{Produit : [fonctionnalité1, fonctionnalité2, …]}``.

    Returns:
        Objet ``pyvis.network.Network`` prêt à être sauvegardé et affiché.
    """
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#343434")
    net.barnes_hut()
    # Ajouter le nœud racine
    net.add_node(company, label=company, color="#E4572E", size=30)
    for product, features in data.items():
        # Ajouter le nœud produit
        net.add_node(product, label=product, color="#4E79A7", size=20)
        net.add_edge(company, product)
        for feature in features:
            # Identifier le nœud de fonctionnalité de manière unique pour éviter
            # d'écraser des nœuds identiques appartenant à différents produits.
            node_id = f"{product}→{feature}"
            net.add_node(node_id, label=feature, color="#76B041", size=15)
            net.add_edge(product, node_id)
    # Options de mise en forme pour un rendu plus agréable
    # Définir les options du graphe. L'activation du mode hiérarchique permet
    # d'obtenir une représentation arborescente claire : le nœud racine en
    # haut, les produits au niveau suivant, puis les fonctionnalités.
    # Préparer la configuration du graphe dans un dictionnaire Python. Elle sera
    # transformée en JSON via ``json.dumps`` afin de garantir que le format est
    # correct pour pyvis (pas de code JavaScript ni de caractère inattendu).
    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "hubsize",
                "levelSeparation": 200,
                "nodeSpacing": 150,
            }
        },
        "nodes": {
            "font": {"size": 14},
            "shape": "box",
        },
        "edges": {
            "color": {"color": "#888888"},
            "arrows": {"to": {"enabled": False}},
            "smooth": {"enabled": False},
        },
        "physics": {"enabled": False},
    }
    net.set_options(json.dumps(options))
    return net


def main() -> None:
    """Fonction principale de l'application Streamlit.

    Elle définit l'interface utilisateur avec deux champs d'entrée et un bouton.
    À l'action du bouton, elle lance la recherche, le scraping et l'affichage du
    graphe interactif.
    """
    st.set_page_config(page_title="Mind‑map Produits et Fonctionnalités", layout="wide")
    st.title("Générateur de Mind‑map des Produits d'une Entreprise")
    st.write(
        "Renseignez le nom de l'entreprise et l'année pour générer une mind‑map des produits et de leurs fonctionnalités."
    )
    # Champs de saisie dans une colonne pour une meilleure présentation
    col1, col2 = st.columns(2)
    with col1:
        company = st.text_input("Nom de l'entreprise", placeholder="Ex.: Apple", key="company")
    with col2:
        year = st.text_input("Année", placeholder="Ex.: 2024", key="year")
    generate = st.button("Générer")
    if generate:
        if not company or not year:
            st.error("Veuillez remplir les deux champs pour lancer la génération.")
        else:
            try:
                with st.spinner("Recherche et extraction en cours…"):
                    product_data = scrape_company_products(company, year, max_results=15)
            except ImportError as e:
                st.error(str(e))
                return
            if not product_data:
                st.warning(
                    "Aucun produit ou fonctionnalité n'a été trouvé.\n"
                    "Vérifiez l'orthographe du nom de l'entreprise et de l'année, ou essayez d'augmenter le nombre de résultats."
                )
            else:
                net = build_mind_map(company, product_data)
                # Sauvegarder le graphe en fichier HTML temporaire
                file_name = f"mindmap_{urllib.parse.quote_plus(company)}_{urllib.parse.quote_plus(year)}.html"
                net.save_graph(file_name)
                # Lire le contenu pour l'afficher dans Streamlit
                with open(file_name, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=750, scrolling=True)
    # Informations supplémentaires dans la barre latérale
    st.sidebar.header("À propos")
    st.sidebar.markdown(
        """
        Cette application effectue un **scraping** sommaire des pages trouvées via
        DuckDuckGo pour identifier les produits et leurs fonctionnalités. Les
        résultats peuvent varier en fonction de la qualité des pages retournées et
        des heuristiques de parsing utilisées.\n\n
        **Dépendances nécessaires :**
        - `streamlit`
        - `beautifulsoup4`
        - `requests`
        - `duckduckgo_search` *(permet la recherche sur DuckDuckGo)*
        - `networkx`
        - `pyvis`
        \n
        Installez-les avec : `pip install streamlit beautifulsoup4 requests duckduckgo_search networkx pyvis`.
        """
    )


if __name__ == "__main__":
    main()
