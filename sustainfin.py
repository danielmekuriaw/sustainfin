from taipy.gui import Gui, notify
import openai
import os
from metaphor_python import Metaphor

METAPHOR_API_KEY = os.environ.get('METAPHOR_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY
metaphor = Metaphor(METAPHOR_API_KEY)

search_query = "Microsoft"
feedback_text = "Feedback"
result_links = ""
news_insights = ""

stat_text = "Stat Text"

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

## Recent News:

<|{result_links}|>

## News Insights:
<|{news_insights}|>

## Statistics:

<|label|{stat_text}}|>

## Your Comments:
<|{feedback_text}|input|label=Your Feedback:|rows=5|>
<|button|on_action=submit_feedback|label=Submit|>

## Comments:

<{comments}>
"""

def search_action(state):
    result_links = ""
    news_insights = ""
    
    # When the user conducts a search, notify them of what they searched for.
    notify(state, message=f"You searched for: {state.search_query}")
    
    USER_QUESTION = f"What's the recent sustainability related news on {search_query} today?"

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

    PROMPT = f"You are a helpful assistant that provides a comprehensive summary with sustainability and investment insights about {search_query} based on the following articles in day-to-day language:\n{combined_articles}"
    USER_QUESTION = "What are the sustainability and investment insights from those articles regarding {search_query}?"

    # Use GPT-3 to summarize the combined content of the articles and extract insights.
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": USER_QUESTION},
            {"role": "user", "content": PROMPT},
        ],
    )
    combined_summary = completion.choices[0].text.strip()
    
    print(combined_summary)
    state.news_insights = combined_summary  # Update the results placeholder with the combined summary.

def submit_feedback(state):
    # Handle feedback submission here
    notify(state, message="Thank you for your feedback!")

# Initial state with placeholders
state = {
    'search_query': '',
    'result_links': 'Placeholder content for links',
    'news_insights': '',
    'stat_text': 'Placeholder content for statistics',
    'feedback_text': '',
    'comments': 'Placeholder for comments',
    'content': 'sustainfin.png'
}

Gui(pages={"/": home_md}).run(state=state)


# Get financial data from either Capital One and/or Bloomberg
# Show financial data using Taipy
# Calculate some interesting statistics and future forecasts and show under the stat section

# Also  rate the sentiments of the news articles - as like a public perception vibe check