# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "html2text",
#     "rich",
#     "beautifulsoup4",
#     "pillow",
#     "google-genai",
#     "httpx",
# ]
# ///

import os
import html2text
import httpx
from bs4 import BeautifulSoup
from PIL import Image
import base64
import mimetypes
import io
from google import genai

def get_image_description(image_path):
    """Get a description of the image using Google GenAI."""
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

    my_file = client.files.upload(file=image_path)

    # with open(os.path.join('pdsaiitm.github.io', image_path), 'rb') as image_file:
    #     image_data = image_file.read()
        

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, "Caption this image."]
    )
    
    #print(response)
    return response.text
    

def convert_html_to_md(html_path,base_dir,output_dir="./markdowns"):
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    md_content = html2text.html2text(str(soup))

    images = soup.find_all('img')
    description_images = """
    """
    for img in images:
        src = img.get('src')
        #print(src)
        
        full_img_path = os.path.join(os.path.dirname(html_path), src)
        description = get_image_description(full_img_path)
        description_images += f"![{description}]({src})\n\n"
        
        # with open(os.path.join(output_dir, 'description.md'), 'w', encoding='utf-8') as md_file:
        #     md_file.write(description)

    md_content += description_images
    # Compute output path while preserving the directory structure
    relpath = os.path.relpath(html_path, base_dir)
    output_md_path = os.path.join(output_dir, os.path.splitext(relpath)[0] + '.md')
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)



    with open(output_md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(md_content)




if __name__ == "__main__":
    os.makedirs('./markdowns', exist_ok=True)
    base_dir = "./pdsaiitm.github.io"
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.html'):
                html_path = os.path.join(root, file)
                convert_html_to_md(html_path,base_dir=base_dir)

    #convert_html_to_md('pdsaiitm.github.io/home.html')
