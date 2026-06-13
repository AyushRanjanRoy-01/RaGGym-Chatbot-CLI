# refer this comprehensive guide for startup the project, execute the following commands in terminal:

# create a virtual environment and install the dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

<!-- # copy the .env.example file to .env
cp .env.example .env -->

# pull the models from ollama
# ollama pull llama3.2:3b
# ollama pull nomic-embed-text

# run this command to ingest the documents
python ingest.py --docs ./docs

# run this command to start the streamlit app
streamlit run app.py

# For better performance, install the Watchdog module:
# xcode-select --install
pip install watchdog