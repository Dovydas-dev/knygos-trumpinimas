import os
from google import genai
import time
from dotenv import load_dotenv
from xai_sdk import Client
import json
import re


load_dotenv(override=True)
client = Client(api_key=os.getenv("XAI_API_KEY"))
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def wait_for_grok_batch(batchID):
    # Paginate through all results
    all_succeeded = []
    all_failed = []
    pagination_token = None
    while True:
        # Fetch a page of results (limit controls page size)
        page = client.batch.list_batch_results(
            batch_id=batchID,
            limit=100,
            pagination_token=pagination_token,
        )
        
        # Collect results from this page
        all_succeeded.extend(page.succeeded)
        all_failed.extend(page.failed)
        
        # Check if there are more pages
        if page.pagination_token is None:
            break
        pagination_token = page.pagination_token
    # Process results - handle different response types
    print(f"Successfully processed: {len(all_succeeded)} requests")
    for result in all_succeeded:
        rid = result.batch_request_id
        resp = result.proto.response
        if resp.HasField("completion_response"):
            # Chat completion response
            print(f"  Tokens used: {result.response.usage.total_tokens}")
    if all_failed:
        print(f"\nFailed: {len(all_failed)} requests")
        for result in all_failed:
            print(f"[{result.batch_request_id}] Error: {result.error_message}")

    return all_succeeded, all_failed

def wait(batchId):
    while True:
        batch = client.batch.get(batch_id=batchId)
        
        pending = batch.state.num_pending
        completed = batch.state.num_success + batch.state.num_error
        total = batch.state.num_requests
        
        print(f"Progress: {completed}/{total} complete, {pending} pending")
        
        if pending == 0:
            print("Batch processing complete!")
            return True
            break
        # Wait before polling again (avoid hammering the API)
        time.sleep(10)


def procces_final_files(all_succeeded, all_failed):
    book_chunks = []
    #print(all_succeeded[0])
    #print(all_succeeded[1])
    sortedResponses = sorted(all_succeeded, key=lambda s: int(re.search(r'\d+$', s.batch_request_id).group()))
    for result in sortedResponses:
        if result.proto.response.HasField("completion_response"):
            book_chunks.append(result.response.content)
    if all_failed:
        for result in all_failed:
            print(f"[{result.batch_request_id}] Error: {result.error_message}")

    
    return book_chunks

def store_all_books():
    batchInfo = json.load(open("batch_ids.json"))

    print("waiting for batches")

    results = {}
    while len(results) < len(batchInfo):
        for batch in batchInfo:
            batchId = batch["batchId"]
            if batchId in results:
                continue
            if wait(batchId):
                succeeded, failed = wait_for_grok_batch(batchId)
                results[batchId] = procces_final_files(succeeded, failed)
    knygos = list(results.values())

    for i, id in enumerate(batchInfo):
        name = id["book_name"]
        knyga = "\n".join(knygos[i])
        knyga = re.sub(r'\n{2,}', '\n', knyga)
        with open(f"Final_output/{name}.txt", "w", encoding="utf-8") as f:
            f.write(knyga)


if __name__ == "__main__":
    store_all_books()
    print("Knygos sutrumpintos!")