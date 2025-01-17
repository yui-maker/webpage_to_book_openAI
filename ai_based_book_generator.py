import os
import requests
import json
from typing import List, Dict
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)

# Verify API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-proj-") or len(api_key) <= 10:
    raise ValueError("Invalid or missing API key.")

# assign which AI model to use
MODEL = "gpt-4o-mini"

# Some websites require proper headers when using them.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    )
}
openai = OpenAI()

# fetching the contents and links from a website.
class Website:
    def __init__(self, url: str):
        self.url = url
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        self.body = response.content

        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        self.text = self._extract_text(soup)
        self.links = self._extract_links(soup)

    @staticmethod
    def _extract_text(soup: BeautifulSoup) -> str:
        """
        Extracts and cleans text from the webpage body.
        """
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            return soup.body.get_text(separator="\n", strip=True)
        else:
            return ""

    @staticmethod
    def _extract_links(soup: BeautifulSoup) -> List[str]:
        """
        Extracts all hyperlinks from the webpage.
        """
        links = [link.get("href") for link in soup.find_all("a")]
        return [link for link in links if link]

    def get_contents(self) -> str:
        """
        Returns the title and text content of the webpage.
        """
        return f"Webpage Title: {self.title}\nWebpage Content:\n{self.text}\n"

LINK_SYSTEM_PROMPT = "you are provided with a list of links found on a webpage. \
    You are able to decide which of the links contain  Python language writing guidlines, \
        such as links about using variables, for loop, while loop etc.\n"
LINK_SYSTEM_PROMPT+= "You should respond in JSON as in this example:"
LINK_SYSTEM_PROMPT+="""
{
    "links": [
    {"type": "learning types of python variables", "url": "https://www.learnpython.org/en/Variables_and_Types"},
    {"type": "learning lists in python", "url": "https://www.learnpython.org/en/Lists"},
    ]
}"""

def generate_user_prompt_for_links(website: Website) -> str:
    """
    Generates a user prompt to identify Python-related links from a website.
    """
    links_list = "\n".join(website.links)
    return (
        f"Here is a list of links on the website {website.url}:\n"
        f"{links_list}\n"
        "Please identify which links are relevant for learning Python programming. "
        "Respond in JSON format with full URLs. Do not include non-Python links such as "
        "About, advertisements, More Languages, discounts, terms of service, support, or privacy policies."
    )


def extract_relevant_links(url: str) -> Dict:
    """
    Fetches a website, identifies relevant links, and returns them in JSON format.
    Includes error handling for API response and JSON decoding.
    """
    try:
        website = Website(url)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": LINK_SYSTEM_PROMPT},
                {"role": "user", "content": generate_user_prompt_for_links(website)},
            ],
        )

        # Ensure the response has the expected structure
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from OpenAI API.")

        cleaned_content = content.lstrip("json").strip()
        return json.loads(cleaned_content)
        
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON. Response: {cleaned_content}")
        return {"links": []}

    except Exception as e:
        print(f"Error extracting relevant links: {e}")
        return {"links": []}



def fetch_all_details(url: str) -> str:
    """
    Gathers details from the landing page and relevant linked pages.
    """
    landing_page = Website(url)
    details = f"Landing Page:\n{landing_page.get_contents()}"
    relevant_links = extract_relevant_links(url)

    for link in relevant_links.get("links", []):
        link_type = link.get("type", "Unknown Type")
        link_url = link.get("url")
        page_details = Website(link_url).get_contents()
        details += f"\n\n{link_type}:\n{page_details}"

    return details

BOOK_SYSTEM_PROMPT = (
    "You are a funny and kind teacher creating engaging and easy-to-understand learning materials "
    "for 11-year-olds to learn Python. Use lots of examples and quizzes."
)

def generate_user_prompt_for_book(course_name: str, url: str) -> str:
    """
    Generates a user prompt for creating Python learning materials.
    """
    details = fetch_all_details(url)
    return (
        f"You are reviewing a Python teaching website called {course_name}.\n"
        f"Here are the contents of its landing page and relevant pages:\n{details[:5_000]}\n"
        "Use this information to create teaching material."
    )


def create_teaching_material(course_name: str, url: str, output_file: str = "teaching_material.md"):
    """
    Creates Python teaching materials using OpenAI and exports the result to a Markdown file.
    
    Args:
        course_name (str): The name of the course or website.
        url (str): The URL of the website to analyze.
        output_file (str): The file to save the results (default: "teaching_material.md").
    """
    try:
        prompt = generate_user_prompt_for_book(course_name, url)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": BOOK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        result = response.choices[0].message.content

        # Save the result to a Markdown file
        export_to_markdown(result, output_file)

    except Exception as e:
        print(f"Error creating teaching material: {e}")


def export_to_markdown(content: str, filename: str = "teaching_material.md"):
    """
    Exports the given content to a Markdown file.
    
    Args:
        content (str): The content to be written to the file.
        filename (str): The name of the output Markdown file (default: "output.md").
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Results successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to markdown: {e}")


# Example Usage
if __name__ == "__main__":
    create_teaching_material("LearnPython", "https://www.learnpython.org/", "python_teaching_material.md")
