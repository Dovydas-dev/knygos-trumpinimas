import os
import time
import glob
from google import genai
import re
import shutil
import unicodedata
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, system
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, SpacyTextSplitter
import pdfplumber
import json


MODEL = "grok-4.20-0309-reasoning"
#MODEL = "grok-4.3"
GEMINI_MODEL = "gemini-3.1-flash-lite"
book_ratio = 40 #What % of the book has to be left

load_dotenv(override=True)
client = Client(api_key=os.getenv("XAI_API_KEY"))
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))




def get_paths():
    pdf_list = glob.glob("input/*.pdf")
    pdf_list = [re.sub(r'\\', '/', pdf) for pdf in pdf_list]
    print(pdf_list[0])
    return pdf_list

def extract_text(pdf_path):
    """Extract all text from a PDF. Returns (full_text, page_count)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages), page_count

def convert_pdfs_to_ascii():
    #delete all of the previous pdf's
    ascii_list = glob.glob("ascii_input/*.pdf")
    for path in ascii_list:
        os.remove(path)
        
    pdf_list = glob.glob("input/*.pdf")
    pdf_list = [re.sub(r'\\', '/', pdf) for pdf in pdf_list]

    for pdf in pdf_list:
        mod_pdf = pdf.replace("input", "ascii_input")
        if os.path.exists(mod_pdf):
            print(f"{mod_pdf} already exists")
        else:
            base = os.path.basename(pdf)
            ascii_base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
            ascii_base = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_base)
            safe_path = os.path.join("ascii_input", ascii_base)
            # Keep original file; just make a copy with safe name
            shutil.copy2(pdf, safe_path)
            print(f"{mod_pdf} created")
    return None

def chunk_book(text):
    chunker = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=200)
    chunks = chunker.split_text(text)
    print(f"Chunks: {len(chunks)}")
    return chunks







def upload_pdfs_to_grok():
    pdfs=glob.glob("ascii_input/*.pdf")
    grok_file_pdf_list = []

    # Upload each PDF
    load_dotenv()
    client = Client(api_key=os.getenv("XAI_API_KEY"))
    for pdf in pdfs:
        file = client.files.upload(pdf, expires_after=82800) #expires after 23 hours
        grok_file_pdf_list.append({"book_name": os.path.basename(pdf), "file": file})

    return grok_file_pdf_list


GROK_SYSTEM = """\
Tu esi lietuvių literatūros teksto atkūrimo redaktorius.

<constraints>
MUST: Naudok tik originalaus failo ir Gemini versijos turinį.
MUST: Galutinio teksto tobulas ilgis — ±{target_len} simbolių.
MUST: Leistinas teksto diapazonas: {lo_len}–{hi_len} simbolių.
MUST: Išlaik autoriaus sakinių ritmą, leksiką ir pasakojimo toną.
MUST: Atkurk viską, kas prarasta — vardus, vietas, datas, dialogo fragmentus, emocines reakcijas, aplinkos detales.
NEVER: Nepridėk informacijos, kurios nėra originale.
NEVER: Grąžink tik tekstą — jokių komentarų, įžangų, paaiškinimų.
NEVER: Nerašyk AI stiliumi — tekstas turi skambėti kaip autorius.
</constraints>

<task>
Vienas praėjimas: surask, ko trūksta Gemini trumpintoje versijoje lyginant su originalu, įterpk į Gemini struktūrą, ištaisyk nenatūralią lietuvių kalbą. Jei reikia jungties su ankstesniu fragmentu — maksimaliai 1–2 frazės iš originalo turinio. Grąžink tik rezultatą.
</task>

<reasoning_discipline>
NEVER: Necituok ir nerašyk pilno teksto reasoning metu.
MUST: Reasoning — tik trūkstamų elementų sąrašas trumpais įrašais (pvz.: "vardas X → įterpti 3 par.", "dialogas → atkurti"). Jokio teksto perrašymo.
MUST: Kuo mažiau reasoning — kuo daugiau tiesiogiai į rezultatą.
</reasoning_discipline>
"""

GROK_USER = """\
<original_file.id>
{file}
</original_file.id>

<gemini_compressed_version>
{compressed}
</gemini_compressed_version>

Atkurk ir patobulink sutrumpintą fragmentą remdamasis originalu. Pateik tik galutinį tekstą.
Galutinis tekstas:\
"""

GEMINI_SYSTEM = """\
Tu esi aukštos kvalifikacijos lietuvių literatūros redaktorius ir teksto trumpintojas.
Tavo užduotis yra sutrumpinti pateiktą knygos fragmentą TIKSLIAI iki {ratio_pct}% jo originalaus simbolių skaičiaus.

SVARBIOS TAISYKLĖS:
1. ILGIO REIKALAVIMAS — PRIVALOMAS: rezultato tekstas turi būti {ratio_pct}% originalaus fragmento ilgio (±5%).
2. Išlaik VISUS esminius siužeto įvykius, pagrindinių veikėjų vystymąsi ir svarbias scenas.
3. Išlaik autoriaus kalbos stilių ir toną kiek įmanoma.
4. Sumažink antrinius aprašymus, pasikartojančias mintis ir per ilgus dialogus.
5. Trumpink, bet NEKURK naujų faktų ar įvykių — tik rinktinai šalink.
6. Rezultatas turi būti sklandžiai skaitomas lietuviškas tekstas.
7. NEANALIZUOK ir NESKAIČIUOK — tiesiog pateik sutrumpintą tekstą be jokių komentarų, skaičiavimų ar įžangų.
8. Nepridėk jokių antraščių ar paaiškinimų — pradėk tiesiogiai nuo teksto.\
"""

GEMINI_USER = """\
Esi teksto redaktorius. Sutrumpink pateiktą knygos fragmentą.

REIKALAVIMAI:
- Tikslinis ilgis: {target_len} simbolių
- Leistinas diapazonas: {lo_len}–{hi_len} simbolių ({ratio_pct}% originalo)
- PRIVALOMA: Prieš rašydamas patikrink savo teksto ilgį. Jei per trumpas — plėsk. Jei per ilgas — trumpink. Kartok kol patenki į diapazoną.
- Išlaikyk originalo stilių, toną ir svarbias detales
- Nekomentuok, nerašyk skaičių ar paaiškinimų

PROCESAS:
1. Apskaičiuok: {target_len} simbolių = maždaug {target_len} / 5 = ~{target_len} žodžių
2. Rašyk sutrumpintą tekstą
3. Įvertink: ar ilgis tarp {lo_len}–{hi_len}? Jei ne — koreguok

--- FRAGMENTAS ---
{chunk}
--- FRAGMENTO PABAIGA ---

Sutrumpintas tekstas (TARP {lo_len} IR {hi_len} SIMBOLIŲ):\
"""





def make_grok_prompt(file_id, gemini_chunk, target_len):
    lo = target_len * .95
    hi = target_len * 1.05

    user_prompt = user(GROK_USER.format(file=file_id, compressed=gemini_chunk))
    system_prompt = system(GROK_SYSTEM.format(target_len=target_len, lo_len=lo, hi_len=hi))

    return system_prompt, user_prompt

def make_batch(model, gemini_chunk, file_id, book_name, request_index, target_len):
    system_prompt, user_prompt = make_grok_prompt(file_id, gemini_chunk, target_len)

    chat = client.chat.create(
        model=model,
        batch_request_id = f"{book_name}-{request_index}"
    )
    chat.append(system_prompt)
    chat.append(user_prompt)

    return chat

def make_request(book_name, file_id, gemini_chunks, chunks):
    batch = client.batch.create(batch_name=book_name)
    batch_id = batch.batch_id
    batch_requests = []

    for i, chunk in enumerate(start=0, iterable=gemini_chunks):
        chunkBatch = make_batch(MODEL, chunk, file_id, book_name, request_index=i+1, target_len=int(len(chunks[i]) * book_ratio / 100))
        batch_requests.append(chunkBatch)

    print(f"Requests: {len(batch_requests)}")
    client.batch.add(batch_id=batch_id, batch_requests=batch_requests)
    return batch_id

def gemini_call(chunk, num):
    target = len(chunk) * book_ratio / 100
    lo = target*0.9
    hi=target*1.1
    system = GEMINI_SYSTEM.format(ratio_pct=book_ratio)
    user = GEMINI_USER.format(chunk=chunk, ratio_pct=book_ratio, target_len=target, lo_len=lo, hi_len=hi)
    print(f"Gemini working on chunk: {num}")

    while True:
        try:
            response = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    {"role": "user", "parts": [{"text": user}]}
                ],
                config={"system_instruction": system}
            )
            return response
        except Exception as e:
            print(f"Gemini error: {e}. Waiting 1 minutes before retry...")
            time.sleep(60)

def shorten_gemini():
    files = glob.glob("ascii_input/*.pdf")
    gemini_books = []
    book_chunks = []

    for file in files:
        converted = ""
        chunk_list = []

        text, pageCount = extract_text(file)
        chunks = chunk_book(text)

        for i, chunk in enumerate(start=1, iterable=chunks):
            response = gemini_call(chunk, i)
            converted += "\n" + response.text
            chunk_list.append(chunk)

        gemini_books.append(converted)
        book_chunks.append(chunk_list)

    return gemini_books, book_chunks


def convert_books_grok():
    #Gets everything thats needed
    gemini_books, book_chunks = shorten_gemini()
    grokFiles = upload_pdfs_to_grok()

    #Rechunks gemini responses
    gemini_chunks = []
    for i in range(len(gemini_books)):
        gemini_chunks.append(chunk_book(gemini_books[i]))

    #Calls Grok 
    batchIds = []
    for i, file in enumerate(grokFiles):
        bookName = file["book_name"]
        file_id = file["file"].id
        batchIds.append(make_request(bookName, file_id, gemini_chunks[i], book_chunks[i]))

    return batchIds, grokFiles


if __name__ == "__main__":
    print("Trumpinamos knygos...")
    convert_pdfs_to_ascii()
    batchIds, grokFiles = convert_books_grok()
    batchInfo = [{"book_name": file["book_name"], "batchId": b_id} for file, b_id in zip(grokFiles, batchIds)]
    #Store Ids on disk
    json.dump(batchInfo, open("batch_ids.json", "w"))
    print("Files are proccessing in the batch!")