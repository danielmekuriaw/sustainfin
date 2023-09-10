from taipy.gui import Gui, notify
import openai
import os
from metaphor_python import Metaphor
import requests
import numpy as np
import pandas as pd
from datetime import datetime

METAPHOR_API_KEY = os.environ.get('METAPHOR_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
FINANCIAL_MODELING_API_KEY = os.environ.get('FINANCIAL_MODELING_API_KEY')

openai.api_key = OPENAI_API_KEY
metaphor = Metaphor(METAPHOR_API_KEY)

search_query = ""
feedback_text = ""
result_links = ""
news_insights = ""

average_environmental_score = ""
average_social_score = ""
average_governance_score  = ""
sustain_fin_score = ""
stat_visualization = ""

start_date = ""
end_date = ""

days_diff = ""

content = "sustainfin.png"

home_md = """

<|{content}|image|>

# Welcome to SustainFin

Explore the financial and sustainability metrics of businesses and investments. 

Enter the name of a business or investment below to get started:

<| |>
<| |>

<|{search_query}|input|label=Search for a business or investment:|>
<|button|on_action=search_action|label=Search|>

<|button|on_action=reset_action|label=Reset|>

## Recent News:

<|{result_links}|>

## News Insights:
<|{news_insights}|>

## Statistics:


<|{start_date}|>


<|{end_date}|>


<|{days_diff}|>


<|{average_environmental_score}|>


<|{average_social_score}|>


<|{average_governance_score}|>


<|{sustain_fin_score}|>


<|{stat_visualization}|>
"""

def search_action(state):
    result_links = ""
    news_insights = ""
    
    # When the user conducts a search, notify them of what they searched for.
    notify(state, message=f"You searched for: {state.search_query}")
    
    stock_symbol = get_stock_symbol(state.search_query)
    
    data = get_esg_data(stock_symbol)
    dates = [entry['date'] for entry in data]
    environmental_scores = [entry['environmentalScore'] for entry in data]
    social_scores = [entry['socialScore'] for entry in data]
    governance_scores = [entry['governanceScore'] for entry in data]
    esg_scores = [entry['ESGScore'] for entry in data]
    
    # Create a pandas dataframe
    df = pd.DataFrame({
        "Dates": dates,
        "Environmental": environmental_scores,
        "Social": social_scores,
        "Governance": governance_scores,
        "ESG": esg_scores
    })
    
    sustainfin_scores = []
    for e, s, g in zip(environmental_scores, social_scores, governance_scores):
        sustainfin_score = 0.4*e + 0.3*s + 0.3*g
        sustainfin_scores.append(sustainfin_score)

    # Now, normalize this score to a range of 0-100, if not already in that range
    max_sf = max(sustainfin_scores)
    min_sf = min(sustainfin_scores)

    normalized_sustainfin_scores = [(score - min_sf) / (max_sf - min_sf) * 100 for score in sustainfin_scores]
    df["SustainFinScore"] = normalized_sustainfin_scores
    
    # Chart properties
    chart_properties = {
        "type": "line",
        "x": "Dates",
        "y[1]": "Environmental",
        "y[2]": "Social",
        "y[3]": "Governance",
        "y[4]": "ESG",
        "y[5]": "SustainFinScore",
        "color[1]": "green",
        "color[2]": "blue",
        "color[3]": "orange",
        "color[4]": "purple",
        "color[5]": "red"
    }

    start = dates[-1]
    state.start_date = f"Start Date: {start}"
    state.end_date = f"End Date: {dates[0]}"
    
    date1 = datetime.strptime(dates[-1], '%Y-%m-%d')
    date2 = datetime.strptime(dates[0], '%Y-%m-%d')
    
    days_difference = (date2 - date1).days
    state.days_diff = f"{days_difference} days"
    
    state.average_environmental_score = np.mean(environmental_scores)
    la = len(environmental_scores)
    state.average_environmental_score = f"Average Environmental Score: {state.average_environmental_score:.3f} ({la} data points)"
    
    state.average_social_score = np.mean(social_scores)
    lb = len(social_scores)
    state.average_social_score = f"Average Social Score: {state.average_social_score:.3f} ({lb} data points)"
    
    state.average_governance_score = np.mean(governance_scores)
    lc = len(governance_scores)
    state.average_governance_score = f"Average Social Score: {state.average_governance_score:.3f} ({lc} data points)"
    
    state.sustain_fin_score = np.mean(normalized_sustainfin_scores)
    state.sustain_fin_score = f"SustainFinScore: {state.sustain_fin_score:.3f}"
    
    # calculate average and provide the start and end date for the collection of the data
    USER_QUESTION = f"What's the recent sustainability related news on {state.search_query} today?"
    SYSTEM_MESSAGE = "You are a helpful assistant that generates search queries based on user questions. Only generate one search query."

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_QUESTION},
        ],
    )

    query = completion.choices[0].message.content
    search_response = metaphor.search(
        query, use_autoprompt=True, start_published_date="2023-09-01"
    )

    # Get top 5 URLs and Titles and format them as Markdown hyperlinks
    top_links_with_titles = [(result.title, result.url) for result in search_response.results][:5]
    state.result_links = "\n".join([f"- [{title}]({url})" for title, url in top_links_with_titles])
    
    # Combine contents of the top 5 articles
    combined_articles = "\n\n".join([result.extract for result in search_response.get_contents().contents[:5]])
    
    PROMPT = f"You are a helpful assistant that provides a comprehensive summary with sustainability and investment insights about {state.search_query} based on the following articles in day-to-day language:\n{combined_articles}"
    USER_QUESTION = "What are the sustainability and investment insights from those articles regarding {state.search_query}?"

    # Use GPT-3 to summarize the combined content of the articles and extract insights.
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": USER_QUESTION},
            {"role": "user", "content": PROMPT},
        ],
    )

    combined_summary = completion.choices[0].message.content
    state.news_insights = combined_summary  # Update the results placeholder with the combined summary.

def submit_feedback(state):
    # Handle feedback submission here
    notify(state, message="Thank you for your feedback!")
    

def reset_action(state):
    result_links = ""
    news_insights = ""
    notify(state, message="Search Reset")

def is_valid_company(query):
    """
    Check if the query is a valid company or investment using Metaphor API.
    Returns True if valid, False otherwise.
    """
    # A simple search on Metaphor
    search_response = metaphor.search(query)
    
    # If we get some results, it might be a valid company.
    if search_response.results:
        return True
    return False

# Define a function to get stock symbol using OpenAI's GPT-4
def get_stock_symbol(company_name):
    # Use OpenAI's API to get the prediction (make sure you have the API key set up)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"The stock exchange symbol for {company_name} is: "},
        ],
        max_tokens = 10
    )
    
    # Return the predicted symbol
    return response.choices[0].message.content

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import certifi
import json
import ssl



def get_esg_data(symbol):
    url = f"https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data?symbol={symbol}&apikey={FINANCIAL_MODELING_API_KEY}"
    # ssl_context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, cafile=certifi.where())    
    data = response.read().decode("utf-8")

    return json.loads(data)


# Initial state with placeholders
state = {
    'search_query': '',
    'result_links': '',
    'news_insights': '',
    'stat_visualization': '',
    'feedback_text': '',
    'comments': 'Placeholder for comments',
    'content': 'sustainfin.png',
    'start_date': '',
    'end_date': '',
    'average_environmental_score': '',
    'average_social_score':'',
    'average_governance_score':'',
}

Gui(pages={"/": home_md}).run(state=state)


# Get financial data from either Capital One and/or Bloomberg
# Show financial data using Taipy
# Calculate some interesting statistics and future forecasts and show under the stat section

# Also  rate the sentiments of the news articles - as like a public perception vibe check