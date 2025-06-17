# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#  ]
# ///


import requests

url = "https://tds-2025t2-vexp-7nhemt9qw-venkatas-projects-862dbda8.vercel.app/api/"
payload = {
    "question": "What is Robin Karp Algorithm?"
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.status_code)
print(response.json)
