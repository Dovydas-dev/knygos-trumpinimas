#!/usr/bin/env python3
"""
Knygos Trumpinimas — compress a Lithuanian book PDF using Gemini + Claude.

Usage:
    python main.py                    # pick a PDF from input/
    python main.py input/book.pdf     # specific file
    python main.py --ratio 0.3        # 30% compression
    python main.py --stage 1          # Gemini only
    python main.py --stage 2          # Claude only (needs cached stage 1)
    python main.py --force            # ignore cache, re-run everything
    python main.py --max-chunks 2     # test with first 2 chunks only
"""
import os
import re
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import anthropic
import click
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from google import genai as google_genai
from google.genai import types
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

load_dotenv()
console = Console()

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-3-flash")
CLAUDE_MODEL       = "claude-opus-4-7"
MAX_RETRIES        = 3
MAX_OUTPUT_TOKENS  = 16_384
TARGET_CHUNK_CHARS = 12_500          # ~5 pages (2 500 chars/page)
FILE_URI_TTL_SECS  = 47 * 3600      # Google Files API keeps uploads for ~48 h

BASE_DIR       = Path(__file__).parent
INPUT_DIR      = BASE_DIR / "input"
GEMINI_OUT_DIR = BASE_DIR / "gemini_output"
CLAUDE_OUT_DIR = BASE_DIR / "claude_output"
CACHE_DIR      = BASE_DIR / "cache"

for _d in (INPUT_DIR, GEMINI_OUT_DIR, CLAUDE_OUT_DIR, CACHE_DIR):
    _d.mkdir(exist_ok=True)

# ── Lithuanian Prompts ────────────────────────────────────────────────────────

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
Sutrumpink toliau pateiktą knygos fragmentą iki {ratio_pct}% jo dydžio. Tai {chunk_num} fragmentas iš {total} iš viso.{continuity}

Tikslinis ilgis: {target_len} simbolių (leistinas diapazonas: {lo_len}–{hi_len}).

Sutrumpink TIK šį fragmentą — pilnas knygos PDF pridėtas kaip kontekstas, kad geriau suprastum siužetą, bet nereikia trumpinti visos knygos.

--- FRAGMENTAS ---
{chunk_text}
--- FRAGMENTO PABAIGA ---

Pateik tik sutrumpintą fragmento tekstą. Jokių komentarų ar skaičiavimų.\
"""

GEMINI_RETRY = """\
Ankstesnis atsakymas buvo {got_len} simbolių, tačiau reikalaujama {lo_len}–{hi_len} simbolių.
{direction_msg}

Bandyk dar kartą. Pateik TIK sutrumpintą tekstą be jokių komentarų ar skaičiavimų.

--- FRAGMENTAS ---
{chunk_text}
--- FRAGMENTO PABAIGA ---\
"""

CLAUDE_SYSTEM = """\
Tu esi aukščiausio lygio lietuvių literatūros redaktorius, specializuojantis sutrumpintos prozos kokybės tobulinimui.

Gausi du tekstus: originalų knygos fragmentą ir Gemini dirbtinio intelekto sutrumpintą jo versiją. Tavo darbas:

1. KALBOS KOKYBĖ: Pataisyk visus nenatūralius sakinius ir mechaniškai nupjautus frazes. Lietuvių kalba turi skambėti natūraliai ir sklandžiai.

2. TURINIO TIKRINIMAS: Naudok originalą kaip nuorodą — jei Gemini išmetė svarbius įvykius ar veikėjų detales, jas grąžink. Jei išlaikė nesvarbiuose dalykuose per daug — galima trumpinti.

3. STILIAUS IŠLAIKYMAS: Išlaik autoriaus balsą — sakinių ritmą, leksiką, emocijų raiškos būdą, kaip matyti originale.

4. NUOSEKLUMAS: Naudok ankstesnio fragmento pabaigą (jei pateikta) kaip kontekstą, kad tekstas sklandžiai tęstųsi.

5. ILGIO KONTROLĖ: Galutinis tekstas turi būti ±10% Gemini versijos ilgio. Tavo darbas — kokybė, ne papildomas trumpinimas.

6. REZULTATAS: Pateik tik pataisytą tekstą. Jokių komentarų ar paaiškinimų. Naudok tik lietuvių kalbą.\
"""

CLAUDE_USER = """\
Originalus knygos fragmentas:

--- ORIGINALAS ---
{original_chunk}
--- ORIGINALO PABAIGA ---
{prev_section}
Gemini sutrumpinta versija (tobulintina):

--- GEMINI VERSIJA ---
{compressed_text}
--- GEMINI VERSIJOS PABAIGA ---

Patobulink Gemini versiją remdamasis originalu. Pateik tik galutinį tekstą.\
"""

CLAUDE_RETRY = """\
Ankstesnis tavo atsakymas buvo {got_len} simbolių, tačiau leidžiamas diapazonas yra {lo_len}–{hi_len} simbolių (±10% nuo Gemini versijos ilgio {input_len} simbolių).
{direction_msg}

Originalus fragmentas (nuorodai):

--- ORIGINALAS ---
{original_chunk}
--- ORIGINALO PABAIGA ---

Gemini versija (tobulintina):

--- GEMINI VERSIJA ---
{compressed_text}
--- GEMINI VERSIJOS PABAIGA ---

Bandyk dar kartą, laikydamasis ilgio reikalavimo. Pateik tik galutinį tekstą.\
"""

# ── PDF & Chunking ────────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> tuple[str, int]:
    """Extract all text from a PDF. Returns (full_text, page_count)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages), page_count


_CHAPTER_RE = re.compile("|".join([
    r"^\s{0,4}(SKYRIUS|Skyrius|skyrius)\s+\d+",
    r"^\s{0,4}\d+\s*\.\s+[A-ZĄČĘĖĮŠŲŪŽ]",
    r"^\s{0,4}[IVX]+\s*\.\s+[A-ZĄČĘĖĮŠŲŪŽ]",
    r"^\s{0,4}(DALIS|Dalis)\s+\d+",
    r"^\s{0,4}(PRIEDAS|Priedas)\b",
]), re.MULTILINE)


def chunk_book(text: str) -> list[str]:
    """Split book into ~5-page chunks, preferring chapter boundaries."""
    positions = [m.start() for m in _CHAPTER_RE.finditer(text)]
    parts = []
    start = 0
    for pos in (positions or []):
        if pos > start and text[start:pos].strip():
            parts.append(text[start:pos].strip())
        start = pos
    if text[start:].strip():
        parts.append(text[start:].strip())
    if not parts:
        parts = [text]

    chunks = []
    for part in parts:
        chunks.extend(_split_by_paragraphs(part) if len(part) > TARGET_CHUNK_CHARS else [part])
    return _merge_short_chunks(chunks)


def _split_by_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        if current and current_len + len(para) > TARGET_CHUNK_CHARS:
            chunks.append("\n\n".join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


def _merge_short_chunks(chunks: list[str]) -> list[str]:
    merged, current_parts, current_len = [], [], 0
    for chunk in chunks:
        if current_parts and current_len + len(chunk) > TARGET_CHUNK_CHARS:
            merged.append("\n\n".join(current_parts))
            current_parts, current_len = [chunk], len(chunk)
        else:
            current_parts.append(chunk)
            current_len += len(chunk)
    if current_parts:
        merged.append("\n\n".join(current_parts))
    return merged

# ── Google Files API ──────────────────────────────────────────────────────────

def get_or_upload_pdf(pdf_path: Path, slug: str, client) -> str:
    """Upload PDF to Google Files API and return its URI (cached for 47 h)."""
    cache_file = CACHE_DIR / slug / "file_uri.txt"
    if cache_file.exists():
        uri, saved_at = cache_file.read_text(encoding="utf-8").splitlines()
        if time.time() - float(saved_at) < FILE_URI_TTL_SECS:
            console.print("  Reusing cached upload")
            return uri

    console.print("  Uploading to Google Files API...")
    # SDK sends the filename as an HTTP header (must be ASCII), so copy to a temp file
    tmp = CACHE_DIR / slug / "_upload.pdf"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(pdf_path.read_bytes())
    try:
        f = client.files.upload(file=tmp, config={"mime_type": "application/pdf", "display_name": slug})
    finally:
        tmp.unlink(missing_ok=True)

    deadline = time.time() + 120
    while "ACTIVE" not in str(f.state).upper():
        if "FAILED" in str(f.state).upper():
            raise RuntimeError(f"Upload failed: {pdf_path.name}")
        if time.time() > deadline:
            raise TimeoutError("Upload timed out after 120 s")
        time.sleep(2)
        f = client.files.get(name=f.name)

    cache_file.write_text(f"{f.uri}\n{time.time()}", encoding="utf-8")
    console.print(f"  Uploaded: {f.name}")
    return f.uri

# ── Stage 1: Gemini Compression ───────────────────────────────────────────────

_REASONING_RE = re.compile(
    r"^\s*(\*[^*]+\*\s*$|.*\d+\s*[+\-]\s*\d+.*=.*|Wait,.*|Let me.*"
    r"|.*\b(chars?|simbolių)\b.*:\s*$|.*Final (text|Polish|version).*)",
    re.IGNORECASE,
)


def run_gemini_chunk(chunk_text, chunk_idx, total_chunks, file_uri, ratio, slug, client, force=False):
    """Compress one chunk with Gemini Flash. Returns (compressed_text, retry_count)."""
    cache_path = CACHE_DIR / slug / f"stage1_chunk_{chunk_idx:03d}.txt"
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8"), 0

    ratio_pct  = int(ratio * 100)
    orig_len   = len(chunk_text)
    target_len = int(orig_len * ratio)
    lo_len     = int(target_len - orig_len * 0.05)
    hi_len     = int(target_len + orig_len * 0.05)
    continuity = " Tęsk tame pačiame stiliuje kaip ankstesni fragmentai." if chunk_idx > 0 else ""

    system = GEMINI_SYSTEM.format(ratio_pct=ratio_pct)
    prompt = GEMINI_USER.format(
        ratio_pct=ratio_pct, chunk_num=chunk_idx + 1, total=total_chunks,
        continuity=continuity, target_len=target_len, lo_len=lo_len, hi_len=hi_len,
        chunk_text=chunk_text,
    )
    result, retries = _call_gemini(client, file_uri, system, prompt), 0

    for _ in range(MAX_RETRIES - 1):
        if lo_len <= len(result) <= hi_len:
            break
        retries += 1
        direction = ("Tekstas per ilgas — reikia labiau trumpinti." if len(result) > hi_len
                     else "Tekstas per trumpas — per daug sutrumpinai, reikia palikti daugiau turinio.")
        result = _call_gemini(client, file_uri, system, GEMINI_RETRY.format(
            got_len=len(result), lo_len=lo_len, hi_len=hi_len,
            direction_msg=direction, chunk_text=chunk_text,
        ))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(result, encoding="utf-8")
    return result, retries


def _call_gemini(client, file_uri, system_instruction, user_prompt):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part(file_data=types.FileData(file_uri=file_uri, mime_type="application/pdf")),
            types.Part(text=user_prompt),
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    lines = response.text.splitlines()
    return "\n".join(ln for ln in lines if not _REASONING_RE.match(ln)).strip()

# ── Stage 2: Claude Refinement ────────────────────────────────────────────────

def run_claude_chunk(compressed_chunk, original_chunk, prev_tail, chunk_idx, slug, client, force=False):
    """Refine one compressed chunk with Claude Opus. Returns (refined_text, retry_count)."""
    cache_path = CACHE_DIR / slug / f"stage2_chunk_{chunk_idx:03d}.txt"
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8"), 0

    input_len = len(compressed_chunk)
    lo_len    = int(input_len * 0.90)
    hi_len    = int(input_len * 1.10)

    prev_section = (
        f"\nAnkstesnio fragmento pabaiga (kontekstui):\n\n"
        f"--- ANKSTESNIO FRAGMENTO PABAIGA ---\n{prev_tail}\n---\n\n"
        if prev_tail else "\n"
    )
    user_content = CLAUDE_USER.format(
        original_chunk=original_chunk,
        prev_section=prev_section,
        compressed_text=compressed_chunk,
    )
    result, retries = _call_claude(client, user_content), 0

    for _ in range(MAX_RETRIES - 1):
        if lo_len <= len(result) <= hi_len:
            break
        retries += 1
        direction = ("Tekstas per ilgas — sumažink jį iki reikiamo ilgio, bet išlaik kokybę."
                     if len(result) > hi_len
                     else "Tekstas per trumpas — išplėsk jį iki reikiamo ilgio, išlaikydamas sklandumą.")
        result = _call_claude(client, CLAUDE_RETRY.format(
            got_len=len(result), lo_len=lo_len, hi_len=hi_len,
            input_len=input_len, direction_msg=direction,
            original_chunk=original_chunk, compressed_text=compressed_chunk,
        ))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(result, encoding="utf-8")
    return result, retries


def _call_claude(client, user_content):
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=[{"type": "text", "text": CLAUDE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


def run_claude_batch(compressed_chunks, original_chunks, slug, client, force):
    """Send all chunks to Claude at once via Batch API (50% cheaper, no inter-chunk continuity)."""
    pending = {
        i: (c, o) for i, (c, o) in enumerate(zip(compressed_chunks, original_chunks))
        if force or not (CACHE_DIR / slug / f"stage2_chunk_{i:03d}.txt").exists()
    }

    if pending:
        requests = [{
            "custom_id": str(i),
            "params": {
                "model": CLAUDE_MODEL,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "system": [{"type": "text", "text": CLAUDE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": CLAUDE_USER.format(
                    original_chunk=original,
                    compressed_text=compressed,
                    prev_section=(
                        f"\nAnkstesnio fragmento pabaiga (kontekstui):\n\n"
                        f"--- ANKSTESNIO FRAGMENTO PABAIGA ---\n{original_chunks[i - 1][-300:]}\n---\n\n"
                        if i > 0 else "\n"
                    ),
                )}],
            },
        } for i, (compressed, original) in pending.items()]

        batch = client.messages.batches.create(requests=requests)
        console.print(f"  Batch submitted: {len(pending)} chunks  (id: {batch.id})")

        while batch.processing_status == "in_progress":
            time.sleep(60)
            batch = client.messages.batches.retrieve(batch.id)
            done = batch.request_counts.succeeded + batch.request_counts.errored
            console.print(f"  Waiting... {done}/{len(pending)} done")

        for result in client.messages.batches.results(batch.id):
            if result.result.type != "succeeded":
                console.print(f"[red]  Chunk {result.custom_id} failed: {result.result.type}[/red]")
                continue
            i    = int(result.custom_id)
            text = result.result.message.content[0].text
            path = CACHE_DIR / slug / f"stage2_chunk_{i:03d}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    return [(CACHE_DIR / slug / f"stage2_chunk_{i:03d}.txt").read_text(encoding="utf-8")
            for i in range(len(compressed_chunks))]

# ── Output Writing ────────────────────────────────────────────────────────────

def write_outputs(text: str, directory: Path, filename: str, title: str) -> None:
    """Write text as both .txt and .docx files."""
    (directory / f"{filename}.txt").write_text(text, encoding="utf-8")
    doc = Document()
    if title:
        doc.add_heading(title, level=1)
    for para in re.split(r"\n{2,}", text):
        if para.strip():
            doc.add_paragraph(para.strip())
    doc.save(str(directory / f"{filename}.docx"))

# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("pdf_path", required=False, type=click.Path(path_type=Path))
@click.option("--ratio",      default=0.5,    show_default=True, help="Compression ratio (0.1–0.9). 0.5 = 50% of original.")
@click.option("--output-dir", default=None,   type=click.Path(path_type=Path), help="Output directory (default: output/)")
@click.option("--force",      is_flag=True,   default=False,     help="Ignore cache, re-run all stages.")
@click.option("--stage",      default="both", show_default=True, type=click.Choice(["1", "2", "both"]), help="Which stage to run.")
@click.option("--max-chunks", default=None,   type=int,          help="Process only first N chunks (useful for testing).")
@click.option("--batch",      is_flag=True,   default=False,     help="Send all Claude chunks as one batch (50% cheaper, slower, no continuity between chunks).")
def compress(pdf_path, ratio, output_dir, force, stage, max_chunks, batch):
    """Compress a Lithuanian book PDF using Gemini Flash + Claude Opus."""

    # Resolve input PDF
    if pdf_path is None:
        pdfs = sorted(INPUT_DIR.glob("*.pdf"))
        if not pdfs:
            console.print("[red]No PDFs in input/. Drop a PDF there and try again.[/red]")
            sys.exit(1)
        if len(pdfs) == 1:
            pdf_path = pdfs[0]
            console.print(f"Using: [bold]{pdf_path.name}[/bold]")
        else:
            console.print("\n[bold]Available PDFs:[/bold]")
            for i, p in enumerate(pdfs, 1):
                console.print(f"  {i}. {p.name}")
            choice = click.prompt("Select a number", type=click.IntRange(1, len(pdfs)), default=1)
            pdf_path = pdfs[choice - 1]
    elif not pdf_path.exists():
        alt = INPUT_DIR / pdf_path
        if alt.exists():
            pdf_path = alt
        else:
            console.print(f"[red]File not found: {pdf_path}[/red]")
            sys.exit(1)

    if output_dir:
        gemini_dir = claude_dir = Path(output_dir)
    else:
        gemini_dir, claude_dir = GEMINI_OUT_DIR, CLAUDE_OUT_DIR
    gemini_dir.mkdir(parents=True, exist_ok=True)
    claude_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^\w-]", "-", pdf_path.stem.lower()).strip("-")
    console.print(f"\n[bold]Book:[/bold]        {pdf_path.name}")
    console.print(f"[bold]Target ratio:[/bold] {int(ratio * 100)}%")
    console.print(f"[bold]Stages:[/bold]       {stage}  {'[yellow](batch)[/yellow]' if batch else ''}")

    with console.status("Extracting text from PDF..."):
        text, page_count = extract_text(pdf_path)
    console.print(f"Extracted [bold]{len(text):,}[/bold] chars from [bold]{page_count}[/bold] pages")

    with console.status("Chunking book..."):
        chunks = chunk_book(text)
    console.print(f"Split into [bold]{len(chunks)}[/bold] chunks (~5 pages each)")

    if max_chunks:
        chunks = chunks[:max_chunks]
        console.print(f"[yellow]Testing mode: processing first {len(chunks)} chunk(s)[/yellow]")

    gemini_key    = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not gemini_key:
        console.print("[red]GEMINI_API_KEY not set in .env[/red]"); sys.exit(1)
    if not anthropic_key:
        console.print("[red]ANTHROPIC_API_KEY not set in .env[/red]"); sys.exit(1)

    gemini_client = google_genai.Client(api_key=gemini_key)
    claude_client = anthropic.Anthropic(api_key=anthropic_key)

    with console.status("Google Files API..."):
        file_uri = get_or_upload_pdf(pdf_path, slug, gemini_client)

    # ── Phase 1: Gemini compression ───────────────────────────────────────────
    compressed_chunks = []

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total} chunks"),
                  console=console, expand=False) as progress:
        task = progress.add_task("[cyan]Gemini compressing...[/cyan]", total=len(chunks))

        for i, chunk in enumerate(chunks):
            if stage in ("1", "both"):
                compressed, retries = run_gemini_chunk(chunk, i, len(chunks), file_uri, ratio, slug, gemini_client, force)
            else:
                cache_path = CACHE_DIR / slug / f"stage1_chunk_{i:03d}.txt"
                if not cache_path.exists():
                    console.print(f"\n[red]Stage 1 cache missing for chunk {i}. Run --stage both first.[/red]")
                    sys.exit(1)
                compressed, retries = cache_path.read_text(encoding="utf-8"), 0

            compressed_chunks.append(compressed)
            orig_len, out_len = len(chunk), len(compressed)
            progress.console.print(
                f"  Chunk {i + 1:>3}: {orig_len:>6,} → {out_len:>6,} chars  ({out_len / max(orig_len, 1) * 100:.1f}%)"
                + (f" [yellow](retried {retries}x)[/yellow]" if retries else "")
            )
            progress.advance(task)

    # ── Phase 2: Claude refinement ────────────────────────────────────────────
    refined_chunks = []

    if stage in ("2", "both"):
        if batch:
            console.print("\n[magenta]Claude batch refinement...[/magenta]")
            refined_chunks = run_claude_batch(compressed_chunks, chunks, slug, claude_client, force)
        else:
            prev_tail = ""
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          BarColumn(), TextColumn("{task.completed}/{task.total} chunks"),
                          console=console, expand=False) as progress:
                task = progress.add_task("[magenta]Claude refining...[/magenta]", total=len(chunks))

                for i, (compressed, chunk) in enumerate(zip(compressed_chunks, chunks)):
                    refined, retries = run_claude_chunk(compressed, chunk, prev_tail, i, slug, claude_client, force)
                    prev_tail = refined[-500:]
                    refined_chunks.append(refined)
                    orig_len, out_len = len(chunk), len(refined)
                    progress.console.print(
                        f"  Chunk {i + 1:>3}: {orig_len:>6,} → {out_len:>6,} chars  ({out_len / max(orig_len, 1) * 100:.1f}%)"
                        + (f" [yellow](retried {retries}x)[/yellow]" if retries else "")
                    )
                    progress.advance(task)
    else:
        refined_chunks = compressed_chunks

    # ── Write outputs ─────────────────────────────────────────────────────────
    book_title = pdf_path.stem.replace("-", " ").replace("_", " ").title()
    filename   = f"{slug}_{int(ratio * 100)}pct"

    console.print("\n[bold green]Done![/bold green]")
    console.print(f"  Original: {len(text):,} chars ({page_count} pages)")

    with console.status("Writing output files..."):
        if stage in ("1", "both"):
            gemini_text = "\n\n".join(compressed_chunks)
            write_outputs(gemini_text, gemini_dir, filename, book_title)
            console.print(f"  Gemini:   {len(gemini_text):,} chars ({len(gemini_text) / max(len(text), 1):.1%})  → {gemini_dir / filename}.txt")
        if stage in ("2", "both"):
            claude_text = "\n\n".join(refined_chunks)
            write_outputs(claude_text, claude_dir, filename, book_title)
            console.print(f"  Claude:   {len(claude_text):,} chars ({len(claude_text) / max(len(text), 1):.1%})  → {claude_dir / filename}.txt")


if __name__ == "__main__":
    compress()
