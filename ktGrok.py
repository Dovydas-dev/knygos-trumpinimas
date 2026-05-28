import json
from kt import upload_pdfs_to_grok, make_request
chunks = json.load(open("book_chunks.json", encoding="utf-8"))
grok_files = upload_pdfs_to_grok()
batch_ids = [make_request(f["book_name"], f["file"].id, chunks[f["book_name"]], chunks[f["book_name"]]) for f in grok_files]
batch_info = [{"book_name": f["book_name"], "batchId": b} for f, b in zip(grok_files, batch_ids)]
json.dump(batch_info, open("batch_ids.json", "w"))
print("Files are processing in the batch!")
