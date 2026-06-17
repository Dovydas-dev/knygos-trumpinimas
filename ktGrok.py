import json
from kt import upload_pdfs_to_grok, make_request, chunk_book_grok, book_ratio

with open("book_chunks.json", "r", encoding="utf-8") as f:
    book_chunks = json.load(f)

grok_files = upload_pdfs_to_grok()

batch_ids = []
for f in grok_files:
    book_name = f["book_name"]
    continuous_text = "\n".join(book_chunks[book_name])
    rechunked = chunk_book_grok(continuous_text)
    batch_ids.append(make_request(book_name, f["file"].id, rechunked, rechunked))

batch_info = [{"book_name": f["book_name"], "batchId": b} for f, b in zip(grok_files, batch_ids)]
json.dump(batch_info, open("batch_ids.json", "w"))
print("Files are processing in the batch!")
