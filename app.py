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

# Importation conditionnelle pour les modules de résumé et d'extraction de mots‑clés.
try:
    # pysummarization pour générer un résumé condensé du texte
    from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
    from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
    from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
    # YAKE pour l'extraction de mots‑clés
    import yake  # type: ignore
    SUMMARY_AVAILABLE = True
except Exception:
    # Si les modules ne sont pas disponibles, les fonctionnalités avancées de
    # résumé/mots clés seront désactivées et le code retournera aux heuristiques
    # simples.
    SUMMARY_AVAILABLE = False

# Importation conditionnelle de spaCy pour le filtrage linguistique. spaCy
# permet de ne conserver que les substantifs ou groupes nominaux (nom d'un
# produit ou d'une fonctionnalité) et d'ignorer les mots inutiles comme les
# verbes ou les mois. Pour utiliser spaCy, vous devrez installer le paquet et
# télécharger les modèles linguistiques (`python -m spacy download fr_core_news_sm` et/ou
# `en_core_web_sm`).
try:
    import spacy  # type: ignore
    try:
        nlp_fr = spacy.load("fr_core_news_sm")  # type: ignore
    except Exception:
        nlp_fr = None  # type: ignore
    try:
        nlp_en = spacy.load("en_core_web_sm")  # type: ignore
    except Exception:
        nlp_en = None  # type: ignore
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    nlp_fr = None
    nlp_en = None

# Définitions de listes d'exclusion pour les mois et quelques mots courants qui
# ne doivent pas être considérés comme des fonctionnalités ou noms de produit.
FRENCH_MONTHS = {
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "août", "aout", "septembre", "octobre", "novembre", "décembre", "decembre"
}
ENGLISH_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"
}
COMMON_EXCLUDE_WORDS = {
    "présentation", "annonce", "nouveau", "nouvelle", "lancement", "update",
    "mise à jour", "release", "présente", "introduit", "introducing", "launch"
}


def _is_valid_feature(candidate: str, company: str, product_name: str) -> bool:
    """Déterminer si une chaîne est un candidat valable pour une fonctionnalité.

    Cette fonction applique différents filtres : suppression des chaînes
    vides, des mois (français et anglais), des mots d'exclusion courants et
    des chaînes contenant le nom de l'entreprise ou du produit. Si spaCy est
    disponible, seuls les tokens ayant une catégorie grammaticale de nom
    (NOUN/PROPN) sont acceptés.

    Args:
        candidate: chaîne à évaluer.
        company: nom de l'entreprise.
        product_name: nom du produit.

    Returns:
        True si la chaîne est jugée valide comme fonctionnalité, False sinon.
    """
    c = candidate.strip()
    if not c:
        return False
    lower = c.lower()
    # Exclure si trop long ou trop court
    if len(c.split()) > 6 or len(c.split()) < 1:
        return False
    # Exclure si contient le nom de l'entreprise ou du produit
    if company.lower() in lower or product_name.lower() in lower:
        return False
    # Exclure si le mot correspond à un mois ou un mot commun
    if lower in FRENCH_MONTHS or lower in ENGLISH_MONTHS or lower in COMMON_EXCLUDE_WORDS:
        return False
    # Utiliser spaCy pour s'assurer qu'au moins un token pertinent est un nom
    if SPACY_AVAILABLE:
        # Choisir le modèle en fonction de la présence d'accents comme proxy
        uses_fr = any(char in "éèàçîôûùëêïùœûæ" for char in lower)
        nlp_model = nlp_fr if uses_fr and nlp_fr is not None else nlp_en
        if nlp_model is not None:
            doc = nlp_model(c)
            # Accepter si le premier token ou au moins un token significatif est un nom/propre
            for token in doc:
                if token.pos_ in {"NOUN", "PROPN", "ADJ"} and not token.is_stop:
                    return True
            return False
    return True

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


def extract_products_features_from_url(url: str, company: str, use_advanced: bool = True) -> Dict[str, List[str]]:
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
                # Filtrer les éléments de liste pour ne conserver que des noms de fonctionnalité pertinents
                if not item:
                    continue
                if not _is_valid_feature(item, company, parent_name):
                    continue
                if item not in product_map[parent_name]:
                    product_map[parent_name].append(item)
        # Chercher des fonctionnalités dans les paragraphes suivants chaque titre
        for heading in soup.find_all(["h1", "h2", "h3"]):
            product_name = heading.get_text(separator=" ").strip()
            if not product_name:
                continue
            # Initialiser la liste si besoin
            product_map.setdefault(product_name, [])
            # Préparer un conteneur pour le texte de la section afin de réaliser
            # éventuellement un résumé et une extraction de mots‑clés.
            section_text_parts: List[str] = []
            # Parcourir les éléments qui suivent ce titre jusqu'au prochain titre de niveau similaire
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h1", "h2", "h3"]:
                    break
                # Rechercher du texte riche en informations potentiellement séparé par des puces, des points ou des tirets
                text = sibling.get_text(separator=" ").strip()
                if not text:
                    continue
                section_text_parts.append(text)
                # Fractionner sur des séparateurs courants (points, virgules, tirets, listes, etc.)
                import re
                candidates = re.split(r"[\n•\-;:\.]", text)
                for cand in candidates:
                    c = cand.strip()
                    # Ne conserver que les fragments valides, filtrés par notre fonction.
                    if not _is_valid_feature(c, company, product_name):
                        continue
                    if c not in product_map[product_name]:
                        product_map[product_name].append(c)
            # Après avoir collecté le texte de la section, appliquer un résumé et une
            # extraction de mots‑clés si les modules nécessaires sont disponibles et si
            # l'utilisateur souhaite activer les fonctionnalités avancées. Cela
            # évite de surcharger l'application lorsqu'``use_advanced`` est False.
            if use_advanced and SUMMARY_AVAILABLE and section_text_parts:
                try:
                    full_section_text = " ".join(section_text_parts)
                    # Limiter la taille du texte à résumer afin d'éviter un calcul trop lourd
                    if len(full_section_text) > 2000:
                        full_section_text = full_section_text[:2000]
                    # Résumé automatique de la section
                    auto_abstractor = AutoAbstractor()
                    auto_abstractor.tokenizable_doc = SimpleTokenizer()
                    auto_abstractor.delimiter_list = [".", "\n"]
                    abstractable_doc = TopNRankAbstractor()
                    summary_dict = auto_abstractor.summarize(full_section_text, abstractable_doc)
                    summary_sentences: List[str] = summary_dict.get("summarize_result", [])  # type: ignore
                    summary_text = " ".join(summary_sentences)
                    # Extraction de mots‑clés avec YAKE
                    language = "fr" if any(
                        char in "éèàçîôûùëêï" for char in summary_text[:100]
                    ) else "en"
                    kw_extractor = yake.KeywordExtractor(lan=language, n=1, top=5)
                    keywords_with_scores = kw_extractor.extract_keywords(summary_text)
                    for kw, score in keywords_with_scores:
                        kw_clean = kw.strip()
                        # Ne conserver que les mots‑clés jugés valides (noms/phrases pertinentes)
                        if not _is_valid_feature(kw_clean, company, product_name):
                            continue
                        if kw_clean not in product_map[product_name]:
                            product_map[product_name].append(kw_clean)
                except Exception:
                    # En cas d'échec du résumé ou de l'extraction, passer
                    pass
        return product_map
    except Exception:
        # En cas d'erreur réseau ou de parsing, retourner un dictionnaire vide.
        return {}


def scrape_company_products(company: str, year: str, max_results: int = 5, use_advanced: bool = True) -> Dict[str, List[str]]:
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
        product_map = extract_products_features_from_url(url, company, use_advanced)
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
    # Options supplémentaires dans la barre latérale
    st.sidebar.subheader("Options avancées")
    use_advanced = st.sidebar.checkbox(
        "Activer l'extraction avancée (résumé et mots‑clés)", value=False,
        help="Désactivez cette option pour accélérer l'extraction et réduire l'utilisation de ressources."
    )
    max_results = st.sidebar.slider(
        "Nombre de pages à analyser", min_value=1, max_value=15, value=5,
        help="Détermine combien de pages web seront analysées pour extraire les produits et fonctionnalités."
    )
    generate = st.button("Générer")
    if generate:
        if not company or not year:
            st.error("Veuillez remplir les deux champs pour lancer la génération.")
        else:
            try:
                with st.spinner("Recherche et extraction en cours…"):
                    product_data = scrape_company_products(company, year, max_results=max_results, use_advanced=use_advanced)
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
        - `pysummarization` *(facultatif : pour résumer les textes et en extraire les points clés)*
        - `yake` *(facultatif : pour l'extraction de mots‑clés)*
        \n
        Installez-les avec : `pip install streamlit beautifulsoup4 requests duckduckgo_search networkx pyvis pysummarization yake`.
        """
    )


if __name__ == "__main__":
    main()
