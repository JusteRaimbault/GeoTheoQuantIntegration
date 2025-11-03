# python3 -m venv .venv
# ./python3.venv/bin/pip install google-cloud-translate
# gcloud auth login
# gcloud config set project $PROJECT (where translate API has been activated on https://console.cloud.google.com/)
# gcloud auth application-default login
from google.cloud import translate_v2 as translate
import pandas,csv

translate_client = translate.Client()

df = pandas.read_csv('corpus_ANNOTATED.csv')

abstracts_list = df['Abstract'].tolist()
titles_list = df['Title'].tolist()

def chunk_list(data_list, chunk_size):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]

BATCH_SIZE = 50

translate_client = translate.Client()

all_translated_text = []
abstract_batches = list(chunk_list(abstracts_list, BATCH_SIZE))

print(f"Total abstracts: {len(abstracts_list)}. Will use {len(abstract_batches)} batches.")

for i, batch in enumerate(abstract_batches):
    print(f"-> Translating Batch {i + 1}/{len(abstract_batches)} (Size: {len(batch)} abstracts)...")

    response = translate_client.translate(
        batch,
        target_language='en',
        source_language='fr'
    )

    translated_text = [translation['translatedText'] for translation in response]
    all_translated_text.extend(translated_text)


df['Abstract'] = all_translated_text

all_translated_titles = []
titles_batches = list(chunk_list(titles_list, BATCH_SIZE))

print(f"Total abstracts: {len(titles_list)}. Will use {len(titles_batches)} batches.")

for i, batch in enumerate(titles_batches):
    print(f"-> Translating Batch {i + 1}/{len(titles_batches)} (Size: {len(batch)} abstracts)...")

    response = translate_client.translate(
        batch,
        target_language='en',
        source_language='fr'
    )

    translated_text = [translation['translatedText'] for translation in response]
    all_translated_titles.extend(translated_text)

df['Title'] = all_translated_titles

df.to_csv('corpus_ANNOTATED_TRANSLATED.csv',index=False, quoting = csv.QUOTE_ALL)
