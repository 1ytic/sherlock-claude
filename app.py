import os
import re
import ast
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from google.cloud import vision
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from ocr import recognize_pdf_content, merge_lines

from timeline import timeline

from geopy.geocoders import GeoNames

from streamlit_agraph import agraph, Node, Edge, Config

from haystack.nodes import PreProcessor, WebRetriever, CohereRanker
from haystack.pipelines import Pipeline
from brave_engine import BraveEngine


def fix_company_name(name):
  name = re.sub(r" (LTD|Ltd|LIMITED|Limited)\.?$", " Ltd.", name.strip())
  name = name.replace("(Public Joint Stock Company)", "JSC")
  return name

def fix_country_name(name):
  name = name.replace("Russian Federation", "Russia")
  name = name.replace("Republic of Cyprus", "Cyprus")
  return name

def fix_company_entities(entities):
  for entity in entities:
    entity["country"] = fix_country_name(entity.get("country", ""))
    entity["name"] = fix_company_name(entity.get("name", ""))
  return entities

def fix_location_entities(entities):
  for entity in entities:
    entity["entity_name"] = fix_company_name(entity["entity_name"])
  return entities

def fix_relationships(relationships):
  result = []
  for relationship in relationships:
    subject = fix_company_name(relationship["subject"])
    object = fix_company_name(relationship["object"])
    if subject == object:
      continue
    predicate = relationship["predicate"]
    if not predicate:
      continue
    result.append({"subject": subject, "predicate": predicate, "object": object})
  return result


@st.cache_data(show_spinner=False)
def trannsform_pdf_to_text(_client, file_name, file_content):
  # Google OCR recognition
  document = recognize_pdf_content(_client, file_content)
  text = ""
  for j, page in enumerate(document):
      text += f"<page{j}>\n"
      text += "\n".join(merge_lines(page.blocks))
      text += f"\n</page{j}>\n"
  return text


@st.cache_data(show_spinner=False)
def extract_entities(_client, documents):
  role_prompt = """You are an investigative journalist proficient in analysing the multipage financial documents to uncover corruption. You are master at extracting information about companies, people, locations, their relationships and related events. Your goal is to extract this information from the documents. Each document begins and ends with <document> and </document> XML tags. A document may have several pages, and each page begins and ends with <page> and </page> XML tags."""

  instruction_prompt = """Please extract the information from the documents in the following steps:
  
1. Extract the information about all companies mentioned in the document. The company may be a business entity, a financial organization or any parties or agents from a contract. For each company try to identify the following details:
- exact name of the company. The commonly used suffixes in company names: Limited, Ltd., LLC, JSC, GmbH, Inc.;
- description of the company, what company does;
- address of the company, where it is registered or headquartered.

2. Extract the information about persons mentioned in the document. For each person try to identify the following details:
- full name of the person;
- description of the person, what person does;
- citizenship of the person.

3. Extract the information about locations mentioned in the document. For each location try to identify the following related details:
- country;
- city;
- postcode;
- address;
- exact entity name related to this location, i.e. the name of the company or person who is registered at this location;
- description what this location is.

4. Extract the information about relationships between the entities. For each relationship try to identify the following details:
- subject of the relationship, a company or a person who initiates the relationship;
- predicate of the relationship, what is the relationship about;
- object of the relationship, a company or a person.

5. Extract the information about events mentioned in the document. For each event try to identify the following details:
- date of the event in YYYY-mm-dd format;
- description what happened.

Additional important rules for the extraction:
* Please make sure that entity names are unique.
* Keep entity even if you have extracted only its name.
* If you can't extract details for the entity or relationship, leave the corresponding field empty.
* Try to extract all possible entities, even if it is not clear whether they are important or not.
* In case of contract or agreement documents, please extract all parties, like borrower, lender, guarantor, chargor, chargee etc.
* If there are multiple relationships between the same subject and object, combine them into one relationship.

Combine all extractions into a single JSON object with the following structure:
{
  "Companies": [
    {"name": "Unknown Ltd.", "description": "", "address": "San Francisco, United States"},
    {"name": "Bank of America Ltd.", "description": "Bank.", "address": "United Kingdom"},
    {"name": "Anthropic", "description": "Company that builds AI tools.", "address": ""}
  ],
  "Persons": [
    {"name": "Albert Einstein", "description": "", "citizenship": "Switzerland"},
    {"name": "Claude Shannon", "description": "American mathematician", "citizenship": "United States"}
  ],
  "Locations": [
    {"country": "", "city": "", "postcode": "", "address": "", "entity_name": "", "description": ""}
  ],
  "Relationships": [
    {"subject": "Unknown Ltd.", "predicate": "owns", "object": "Bank Ltd."}
  ],
  "Events": [
    {"date": "YYYY-mm-dd", "description": ""}
  ]
}

Please, do not output preamble before or conclusion after the list."""
  completion = _client.completions.create(
    model="claude-2",
    temperature=0,
    max_tokens_to_sample=4000,
    prompt=f"{HUMAN_PROMPT} {role_prompt}\n\n{instruction_prompt}\n\nHere are the documents, in <document></document> XML tags:\n{documents}{AI_PROMPT}\n{{",
  )
  try:
    result = ast.literal_eval("{" + completion.completion)
    if "Companies" in result:
      result["Companies"] = fix_company_entities(result["Companies"])
    if "Locations" in result:
      result["Locations"] = fix_location_entities(result["Locations"])
    if "Relationships" in result:
      result["Relationships"] = fix_relationships(result["Relationships"])
  except:
    st.error(completion.completion)
    result = {}
  return result


@st.cache_data(show_spinner=False)
def combine_entities(_client, documents):
  role_prompt = """You are an investigative journalist proficient in analysing the financial information to uncover corruption. You are master at aggregating and merging information about many different companies, people, locations, their relationships and related events. Your goal is to remove duplicates from the input data. The input data presented inside <data></data> XML tags and consists of the lists for each type of entity: <companies>, <persons>, <locations>, <relationships> and <events>."""

  instruction_prompt = """Please remove duplicates from the input data in the following steps:
  
1. Merge the information about similar companies. The company may be a business entity, a financial organization or any parties or agents from a contract. For the similar companies try to merge the following details:
- description of the company, what company does;
- address of the company, where it is registered or headquartered.

2. Merge the information about similar persons. For the similar persons try to merge the following details:
- description of the person, what person does;
- citizenship of the person.

3. Merge the information about similar locations mentioned in the document. For each location try to merge the following related details:
- country;
- city;
- postcode;
- address;
- exact entity name related to this location, i.e. the name of the company or person who is registered at this location;
- description what this location is.

4. Merge the information about similar relationships between the entities. For the similar relationships try to merge the following details:
- subject of the relationship, a company or a person who initiates the relationship;
- predicate of the relationship, what is the relationship about;
- object of the relationship, a company or a person.

5. Merge the information about similar events. For the similar event try to merge the following details:
- date of the event in YYYY-mm-dd format;
- description what happened.

Use the following important rules to rename companies only in specific cases:
* If the company name includes "LTD" rename it to "Ltd.".
* If the company name includes "Limited" rename it to "Ltd.".
* If the company name includes "Incorporated" rename it to "Inc.".
* If the company name includes "Public Joint Stock Company" rename it to "JSC".
* If the company name includes "Limited Liability Company" rename it to "LLC".
* LLC abbreviation must be followed by the company name. For example, "LLC Unknown" must be renamed to "Unknown LLC".
* Consider PJSC and JSC abbreviations as the same. For example, "PJSC Unknown" and "Unknown JSC" must be merged into one entity "Unknown JSC".

Use the following rules for the merging similar entities and removing duplicates:
* Make sure that output entity names are unique.
* Entities considered similar if they have similar names.
* Try to use abbreviations as much as possible. For example, "Limited Liability Company Unknown" and "Unknown LLC" must be merged into one company "Unknown LLC".
* Entity names are not case sensitive. For example, "Unknown Bank Ltd." and "UNKNOWN BANK LIMITED" must be merged into one entity "Unknown Bank Ltd.".
* Do not trim or cut the original entity names. Always choose the longest original name.
* If two entities have similar names, but different details, merge them into one entity with the combined details.
* If you can't merge or don't know the details for the entity or relationship, leave the corresponding field empty.
* Make sure that the subject and object of the relationship refer to the existing entities.
* If the subject and the object of the extracted relationship refer to the same entity, remove the relationship.

Finally, output summary of extracted and merged information into a single paragraph.

Merge the input data into a single JSON object with the following structure:
{
  "Companies": [
    {"name": "Unknown Ltd.", "description": "", "address": "San Francisco, United States"},
    {"name": "Bank of America Ltd.", "description": "Bank.", "address": "United Kingdom"},
    {"name": "Anthropic", "description": "Company that builds AI tools.", "address": ""}
  ],
  "Persons": [
    {"name": "Albert Einstein", "description": "", "citizenship": "Switzerland"},
    {"name": "Claude Shannon", "description": "American mathematician", "citizenship": "United States"}
  ],
  "Locations": [
    {"country": "", "city": "", "postcode": "", "address": "", "entity_name": "", "description": ""}
  ],
  "Relationships": [
    {"subject": "Unknown Ltd.", "predicate": "owns", "object": "Bank Ltd."}
  ],
  "Events": [
    {"date": "YYYY-mm-dd", "description": ""}
  ],
  "Summary": ""
}

For example, the following input data:

<data>
<companies>
<company>
<name>PJSC Unknown</name>
<description></description>
<address></address>
</company>
<company>
<name>UNKNOWN JSC</name>
<description></description>
<address>San Francisco, United States</address>
</company>
</companies>
</data>

must be merged into the following JSON object:

{
  "Companies": [
    {"name": "Unknown JSC", "description": "", "address": "San Francisco, United States"},
  ]
}

Please, do not output preamble before or conclusion after the list."""

  inputs = convert_entities_to_xml(documents)

  completion = _client.completions.create(
    model="claude-2",
    temperature=0,
    max_tokens_to_sample=6000,
    prompt=f"{HUMAN_PROMPT} {role_prompt}\n\n{instruction_prompt}\n\nHere are the input data, in <data></data> XML tags:\n{inputs}{AI_PROMPT}\n{{",
  )
  try:
    result = ast.literal_eval("{" + completion.completion)
    if "Companies" in result:
      result["Companies"] = fix_company_entities(result["Companies"])
    if "Locations" in result:
      result["Locations"] = fix_location_entities(result["Locations"])
    if "Relationships" in result:
      result["Relationships"] = fix_relationships(result["Relationships"])
  except:
    st.error(completion.completion)
    result = {}
  return result


@st.cache_data(show_spinner=False)
def answer_question(_client, documents, entities, question):

  input_documents = "<documents>\n"
  for i, d in enumerate(documents):
    input_documents += f"<document{i}>\n{d}\n</document{i}>\n"
  input_documents += "</documents>\n"

  input_entities = convert_entities_to_xml(entities)

  role_prompt = """You are an investigative journalist proficient in analysing the financial documents to uncover corruption. You are master at aggregating and merging extracted information about many different companies, people, locations, their relationships and related events. Your goal is to answer question about extracted information. The extracted data presented inside <data></data> XML tags and consists of the lists for each type of entity: <companies>, <persons>, <locations>, <relationships> and <events>. Additionally, you can use the original text of documents inside <document></document> and <page></page> XML tags."""

  completion = _client.completions.create(
    model="claude-2",
    temperature=0,
    max_tokens_to_sample=6000,
    prompt=f"{HUMAN_PROMPT} {role_prompt}\n\nHere are the documents, in <document></document> XML tags:\n{input_documents}\n\nHere are the extracted data, in <data></data> XML tags:\n\n{input_entities}\n\nPlease answer the following question: {question}\n\n{AI_PROMPT}",
  )
  
  return completion.completion


def convert_entities_to_xml(documents):

  companies = []
  persons = []
  locations = []
  relationships = []
  events = []
  for document in documents:
    companies += document.get("Companies", [])
    persons += document.get("Persons", [])
    locations += document.get("Locations", [])
    relationships += document.get("Relationships", [])
    events += document.get("Events", [])

  companies = sorted(companies, key=lambda e: e.get("name", ""))
  persons = sorted(persons, key=lambda e: e.get("name", ""))
  locations = sorted(locations, key=lambda e: e.get("entity_name", ""))
  relationships = sorted(relationships, key=lambda e: e.get("subject", ""))
  events = sorted(events, key=lambda e: e.get("date", ""))

  inputs = "<data>\n<companies>\n"
  for e in companies:
    inputs += f"""<company>
<name>{e.get('name')}</name>
<description>{e.get('description')}</description>
<address>{e.get('address')}</address>
</company>
"""
  inputs += "</companies>\n"

  inputs += "<persons>\n"
  for e in persons:
    inputs += f"""<person>
<name>{e.get('name')}</name>
<description>{e.get('description')}</description>
<citizenship>{e.get('citizenship')}</citizenship>
</person>
"""
  inputs += "</persons>\n"

  inputs += "<locations>\n"
  for e in locations:
    inputs += f"""<location>
<country>{e.get('country')}</country>
<city>{e.get('city')}</city>
<postcode>{e.get('postcode')}</postcode>
<address>{e.get('address')}</address>
<entity_name>{e.get('entity_name')}</entity_name>
<description>{e.get('description')}</description>
</location>
"""
  inputs += "</locations>\n"

  inputs += "<relationships>\n"
  for e in relationships:
    inputs += f"""<relationship>
<subject>{e.get('subject')}</subject>
<predicate>{e.get('predicate')}</predicate>
<object>{e.get('object')}</object>
</relationship>
"""
  inputs += "</relationships>\n"

  inputs += "<events>\n"
  for e in events:
    inputs += f"""<event>
<date>{e.get('date')}</date>
<description>{e.get('description')}</description>
</event>
"""
  inputs += "</events>\n"

  inputs += "</data>\n\n"
  return inputs


@st.cache_data(show_spinner=False)
def retrieve_web_passages(_client, query):
  response = _client.run(query)
  return response.get("documents", [])


@st.cache_data(show_spinner=False)
def rerank_web_passages(_client, query, passages):
  response = _client.run(query=query, documents=passages)
  return response[0].get("documents", [])


@st.cache_data(show_spinner=False)
def country_name_to_image(name):
  name = name.lower()
  names_mapping = {
    "cyprus": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTB2ZtaOP1ziqo2vt-CNaFtSXx_EJBDOKG6g-ar2EKZrIbPjiaAPOXqCQ&s=0",
    "republic of cyprus": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTB2ZtaOP1ziqo2vt-CNaFtSXx_EJBDOKG6g-ar2EKZrIbPjiaAPOXqCQ&s=0",
    "british virgin islands": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-byRgXVzdXh0FaRvqd0P1FI3pjSNG__YsAEpzAGWXE0mErouVHjadcvw&s=0",
    "united kingdom": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT5gSIvqufz2nI-4_jo21XRhh91Asvc9U-RId0_Aa1AOghdZlC520dkesY&s=0",
    "russia": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS-EvHOipeEzsqUKnEJjWUkDwH0Pv-uILepyFFhD9dPwQcyY9lk25SC0g&s=0",
    "russian federation": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS-EvHOipeEzsqUKnEJjWUkDwH0Pv-uILepyFFhD9dPwQcyY9lk25SC0g&s=0",
  }
  default_image = "https://img.icons8.com/?size=160&id=1ILfgmtuhsde&format=png"
  if name in names_mapping:
    return names_mapping[name]
  # TODO: use API to get country flag
  return default_image


load_dotenv()  # take environment variables from .env.

vision_client = vision.ImageAnnotatorClient()

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

web_preprocessor = PreProcessor(
    progress_bar=False,
    remove_substrings=["[ edit ]"],
    split_length=500,
)

web_retriever = WebRetriever(
    api_key=os.environ["SERPER_API_KEY"],
    search_engine_provider=BraveEngine(os.environ["BRAVE_API_KEY"]),
    top_search_results=5,
    preprocessor=web_preprocessor,
    mode="preprocessed_documents",
    top_k=5,
)

web_ranker = CohereRanker(
    api_key=os.environ["COHERE_API_KEY"],
    model_name_or_path="rerank-english-v2.0",
)

web_pipeline = Pipeline()
web_pipeline.add_node(component=web_retriever, name="Retriever", inputs=["Query"])
web_pipeline.add_node(component=web_ranker, name="Ranker", inputs=["Retriever"])

st.set_page_config(
    page_title="Documents Mining Tool",
    page_icon="⛏️",
)

st.title("Documents Mining Tool Investigative Journalists")

# Initialize analysis history
if "documents" not in st.session_state:
    st.session_state.documents = []
    st.session_state.entities = []
    st.session_state.companies = []
    st.session_state.persons = []
    st.session_state.relations = []
    st.session_state.events = []
    st.session_state.locations = []
    st.session_state.messages = []
    st.session_state.contexts = []

uploaded_files = st.file_uploader(
  "",
  accept_multiple_files=True,
  type=["pdf"], # , "png", "jpg", "jpeg"
)

if uploaded_files:

  with st.status("Processing documents...", expanded=True) as status:

    st.session_state.documents = []
    st.session_state.entities = []

    sorted_uploaded_files = sorted(uploaded_files, key=lambda f: f.name)

    for i, file in enumerate(sorted_uploaded_files):

      # Stage 1: OCR recognition
      st.write(f"Recognize #{i+1}: {file.name}")

      document = trannsform_pdf_to_text(
        vision_client,
        file.name,
        file.getvalue(),
      )

      # Stage 2: NLP analysis
      st.write(f"Extract #{i+1}: {file.name}")

      document = document.strip()
      st.session_state.documents.append(document)

      entities = extract_entities(anthropic_client, f"<document>\n{document}\n</document>\n\n")
      st.session_state.entities.append(entities)

    st.write("Combine extraction...")
    llm_result = combine_entities(anthropic_client, st.session_state.entities)

    # Stage 3: Web search
    contexts = []
    queries = []
    if "Events" in llm_result:
      st.write(f"Search web for Events ...")
      for entity in llm_result["Events"]:
        query = entity.get("description")
        if query:
          try:
            contexts.extend(retrieve_web_passages(web_pipeline, query))
            queries.append(query)
          except:
            continue
    if "Relationships" in llm_result:
      st.write(f"Search web for Relationships ...")
      for entity in llm_result["Relationships"]:
        query = entity.get("subject", "") + " " + entity.get("predicate", "") + " " + entity.get("object", "")
        if query:
          try:
            contexts.extend(retrieve_web_passages(web_pipeline, query))
            queries.append(query)
          except:
            continue
    
    # Stage 3: Rerank web results
    query = llm_result.get("Summary", ", ".join(queries))
    contexts = rerank_web_passages(web_ranker, query, contexts)

    st.session_state.companies = llm_result["Companies"] if "Companies" in llm_result else []
    st.session_state.persons = llm_result["Persons"] if "Persons" in llm_result else []
    st.session_state.relations = llm_result["Relationships"] if "Relationships" in llm_result else []
    st.session_state.events = llm_result["Events"] if "Events" in llm_result else []
    st.session_state.locations = llm_result["Locations"] if "Locations" in llm_result else []
    st.session_state.contexts = contexts[:3]

    if "Summary" in llm_result and len(st.session_state.messages) == 0:
      st.session_state.messages.append({"role": "assistant", "content": llm_result["Summary"]})
    
    status.update(label="Initial analysis completed", state="complete", expanded=False)

  # for i, (doc_name, doc_content) in enumerate(st.session_state.documents.items()):
  #   st.text_area("Document", doc_content, key=f"doc:{i}", height=175)

#############################################################
# Locations

if len(st.session_state.locations):

  st.header(f"📍 Locations ({len(st.session_state.locations)})")

  tab1, tab2 = st.tabs(["Map", "Tabular"])

  with tab1:

    geo = GeoNames(username="sorokin")

    data = []
    for loc in st.session_state.locations:
      point = geo.geocode(query=loc.get("city", "") + ", " + loc.get("country", ""), exactly_one=True)
      if point:
        data.append([point.latitude, point.longitude])
    df = pd.DataFrame(data, columns=["latitude", "longitude"])

    # Fix map width issue (hopefully)
    # https://github.com/gee-community/geemap/issues/713
    make_map_responsive= """
    <style>
    [title~="st.iframe"] { width: 100%}
    </style>
    """
    st.markdown(make_map_responsive, unsafe_allow_html=True)

    st.map(df, use_container_width=True)

  with tab2:
    st.dataframe(st.session_state.locations, use_container_width=True)

#############################################################
# Events

if len(st.session_state.events):

  st.header(f"📅 Events ({len(st.session_state.events)})")

  tab1, tab2 = st.tabs(["Timeline", "Tabular"])

  with tab1:
    data = []
    for event in st.session_state.events:
      parts = event["date"].split("-")
      if len(parts) == 3:
        start_date = {"year": parts[0], "month": parts[1], "day": parts[2]}
      elif len(parts) == 2:
        start_date = {"year": parts[0], "month": parts[1]}
      elif len(parts) == 1:
        start_date = {"year": parts[0]}
      else:
        continue
      data.append({
        "start_date": start_date,
        "text": {"text": event["description"]},
      })
    timeline({"events": data}, height=330)

  with tab2:
    st.dataframe(st.session_state.events, use_container_width=True)

#############################################################
# Entities

num_entities = len(st.session_state.companies) + len(st.session_state.persons)

if num_entities:

  st.header(f"👨‍👨‍👧‍👦 Entities ({num_entities})")

  tab1, tab2 = st.tabs(["Companies", "Persons"])

  with tab1:
    st.dataframe(st.session_state.companies, use_container_width=True)

  with tab2:
    st.dataframe(st.session_state.persons, use_container_width=True)

#############################################################
# Relations

if len(st.session_state.relations):

  st.header(f"🔗 Relations ({len(st.session_state.relations)})")

  # https://blog.streamlit.io/the-streamlit-agraph-component/

  config = Config(
    height=500,
    directed=False,
    physics=False,
    interaction={
      "zoomView": False,
      "selectable": False,
    },
  )

  # https://visjs.github.io/vis-network/docs/network/nodes.html

  tab1, tab2, tab3 = st.tabs(["Graph", "Tabular", "Meme"])

  with tab1:
    unique_companies = set([c["name"].lower() for c in st.session_state.companies])
    unique_persons = set([p["name"].lower() for p in st.session_state.persons])
    countries_mapping = {l["entity_name"]: l["country"] for l in st.session_state.locations if l["country"]}
    location_entities = sorted(set(l["entity_name"] for l in st.session_state.locations if l["entity_name"]))
    countries = sorted(set(l["country"] for l in st.session_state.locations if l["country"]))

    def entity_name_to_image(name):
      name = name.lower()
      if name == "konstantin ernst":
        image = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSivDz05OV9HPfAIvom7AxLn8mz0m0kAhEolqfRcg&s=0"
      elif name == "rcb bank ltd.":
        image = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScDx05_qYIc9Uz-NIuGUPmYcwQB1ngsvOZQV-9V08GrapHrKJHKiH77Q&s=0"
      elif name == "jsc vtb bank" or name == "vtb bank jsc":
        image = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRINXcqA36Y-jT3Y1QohvE1JnbP2nFe4sjMCgi5VJBvr8i3H6erKKJ2nA&s=0"
      elif name in unique_persons:
        image = "https://img.icons8.com/?size=2x&id=7819&format=png"
      elif name in unique_companies:
        image = "https://img.icons8.com/?size=160&id=pRCb69cRQQiq&format=png"
      elif name in countries:
        image = "https://img.icons8.com/?size=160&id=1ILfgmtuhsde&format=png"
      else:
        image = "https://img.icons8.com/?size=160&id=pRCb69cRQQiq&format=png"
      return image

    labels = set(location_entities)
    for r in st.session_state.relations:
      labels.add(r.get("subject"))
      labels.add(r.get("object"))
    
    nodes_names = set()
    nodes = []
    edges = []
    for label in countries:
      if label in nodes_names:
        continue
      image = country_name_to_image(label)
      nodes.append(Node(id=label, label=label, size=10, shape="image", image=image))
      nodes_names.add(label)
    for label in sorted(labels):
      if label in nodes_names:
        continue
      image = entity_name_to_image(label)
      if label in countries_mapping:
        edges.append(Edge(source=label, target=countries_mapping[label]))
      nodes.append(Node(id=label, label=label, size=30, shape="image", image=image))
      nodes_names.add(label)
    for r in st.session_state.relations:
      edges.append(Edge(source=r.get("subject"), target=r.get("object")))

    for p in st.session_state.persons:
      citizenship = p.get("citizenship")
      if citizenship:
        edges.append(Edge(source=p.get("name"), target=citizenship))
    
    agraph(nodes, edges, config)

  with tab2:
    st.dataframe(st.session_state.relations, use_container_width=True)

  with tab3:
    st.image("https://i.kym-cdn.com/photos/images/original/002/546/187/fb1.jpg", use_column_width=True)

#############################################################
# Contexts

if len(st.session_state.contexts):

  st.header(f"📚 Contexts ({len(st.session_state.contexts)})")

  for i, d in enumerate(st.session_state.contexts):
    t = st.text_input("Source", d.meta["url"], key=f"url:{i}")
    txt = st.text_area("Passage", d.content, key=f"text:{i}", height=175)

#############################################################
# Answers

if len(st.session_state.documents):

  st.header(f"🙋 Answers ({len(st.session_state.messages)})")

  # Display chat messages from history on app rerun
  for message in st.session_state.messages:
      with st.chat_message(message["role"], avatar="🕵️‍♂️" if message["role"] == "user" else "🤖"):
          st.markdown(message["content"])

  # React to user input
  if prompt := st.chat_input("Ask me a question"):
      # Display user message in chat message container
      st.chat_message("user", avatar="🕵️‍♂️").markdown(prompt)
      # Add user message to chat history
      st.session_state.messages.append({"role": "user", "content": prompt})

      response = answer_question(anthropic_client, st.session_state.documents, st.session_state.entities, prompt)
      # Display assistant response in chat message container
      with st.chat_message("assistant", avatar="🤖"):
          st.markdown(response)
      # Add assistant response to chat history
      st.session_state.messages.append({"role": "assistant", "content": response})
