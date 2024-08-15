import os
from embedchain import App

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
app = App.from_config(config_path="save_db_config.yaml")

app.add("https://mor.org/about", data_type="web_page")
app.add("https://mor.org/blog/morpheus-digest-3", data_type="web_page")
app.add("https://mor.org/blog/mor-token-claim-on-may-8th-faq", data_type="web_page")
app.add("https://mor.org/blog/morpheus-digest-2", data_type="web_page")
app.add("https://mor.org/blog/morpheus-weekly-digest-1", data_type="web_page")
app.add("https://mor.org/blog/mor-claim-guide-interacting-directly-with-the-smart-contracts", data_type="web_page")
app.add("https://mor.org/blog/10-reasons-to-be-excited-about-morpheus", data_type="web_page")
app.add("https://mor.org/blog/morpheus-coder-guide-and-best-practices", data_type="web_page")
app.add("https://mor.org/blog/morpheus-beliefs", data_type="web_page")
app.add("https://mor.org/faqs", data_type="web_page")


