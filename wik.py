import wikipedia
from concurrent.futures import ThreadPoolExecutor
import re

def get_wiki_pages(text):
    results = wikipedia.search(text)
    with ThreadPoolExecutor() as executor:
        pages = executor.map(get_page, results)
    return list(filter(None, pages))


def get_page(name):
    try:
        page = wikipedia.page(name)
        return split_page(page.content)#.split('\n\n')
    except wikipedia.exceptions.DisambiguationError as e:
        # You can handle disambiguation errors here if needed
        # For simplicity, we're just returning the first option's content
        try:
            return split_page(wikipedia.page(e.options[0]).content)#.split('\n\n')
        except:
            return None
    except Exception as e:
        print(e)
        return None

def split_page(content):
    # Use regex to find section titles
    section_splits = re.split(r'(== [^=]+ ==)', content)

    # Filter out any empty segments
    segments = [segment.strip() for segment in section_splits if segment.strip()]

    # If there are no sections, return the entire content
    if len(segments) == 1:
        return segments

    # Combine section titles with their content
    combined_segments = []
    i = 0
    while i < len(segments):
        # If current segment is a section title
        if segments[i].startswith("==") and segments[i].endswith("=="):
            # If next segment is also a section title or we're at the end, just add the title
            if i == len(segments) - 1 or (segments[i+1].startswith("==") and segments[i+1].endswith("==")):
                combined_segments.append(segments[i])
                i += 1
            else:
                combined_segments.append(segments[i] + "\n" + segments[i+1])
                i += 2
        else:
            combined_segments.append(segments[i])
            i += 1

    return combined_segments


if __name__ == "__main__":
    results = get_wiki_pages("Python programming")
    for content in results:
        print("!!!new page!!!")
        print("\n!!!\n".join([x for x in content]))  # Printing the first 500 characters for brevity
