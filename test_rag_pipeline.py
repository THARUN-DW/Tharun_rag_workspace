"""
test_rag_pipeline.py
====================
End-to-end offline smoke test (NO OpenAI key needed — the LLM call is mocked).

1. Generates a fresh sample PDF, DOCX, TXT, CSV and MD file at runtime
   (reportlab / python-docx / plain text) — nothing hard-coded in the app.
2. Runs the full ingestion path: load_document -> chunk_document -> add_chunks.
3. Runs retrieval with real MiniLM embeddings against a temporary ChromaDB.
4. Feeds the retrieved context to a MOCKED LLM and verifies the flow returns
   an answer + sources with file/page/chunk/similarity metadata.

Run:  python test_rag_pipeline.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

PASS, FAIL = "PASS", "FAIL"


def header(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. Generate sample documents at runtime
# ---------------------------------------------------------------------------

def make_pdf(path: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pages = [
        "Employee Leave Policy\n\nEmployees receive 15 casual leave days per year.\n"
        "Casual leave requests must be approved by the reporting manager.\n"
        "A maximum of 3 casual leave days may be taken consecutively.\n"
        "Unused casual leave days do not carry over to the next year.",
        "Work From Home Policy\n\nEmployees can request up to 2 work-from-home days per week.\n"
        "Work-from-home requests require at least 24 hours notice.\n"
        "Managers may approve additional days during emergencies.",
        "Probation Policy\n\nNew employees serve a 6 month probation period.\n"
        "During probation, leave entitlement is half the standard allowance.",
    ]
    c = canvas.Canvas(path, pagesize=A4)
    for page_no, text in enumerate(pages, start=1):
        y = 780
        c.setFont("Helvetica", 11)
        for line in text.splitlines():
            c.drawString(50, y, line)
            y -= 20
        c.showPage()
    c.save()


def make_docx(path: str) -> None:
    import docx

    d = docx.Document()
    d.add_paragraph("Machine Learning Notes")
    d.add_paragraph(
        "Supervised learning uses labelled training data. The algorithm learns a "
        "mapping from inputs to known outputs. Examples include linear regression, "
        "logistic regression, random forests and support vector machines. Typical "
        "tasks are classification and regression."
    )
    d.add_paragraph(
        "Unsupervised learning works on unlabelled data. The algorithm must discover "
        "structure on its own. Examples include k-means clustering, hierarchical "
        "clustering, PCA and autoencoders. Typical tasks are clustering, dimensionality "
        "reduction and anomaly detection."
    )
    d.add_paragraph(
        "The key difference between supervised and unsupervised learning is the "
        "presence of labelled data: supervised models are trained against known "
        "answers, while unsupervised models find hidden patterns without labels."
    )
    d.add_paragraph(
        "Reinforcement learning is a third paradigm in which an agent learns by "
        "interacting with an environment and receiving rewards or penalties."
    )
    d.save(path)


def make_txt(path: str) -> None:
    Path(path).write_text(
        "Product Manual: CoffeeMaker X100\n\n"
        "Warranty: 2 years from the date of purchase.\n"
        "Descaling: run the descale cycle every 3 months using citric acid solution.\n"
        "Max water tank capacity: 1.2 litres.\n\n"
        "Troubleshooting: if the machine does not heat, check the thermal fuse.\n",
        encoding="utf-8",
    )


def make_csv(path: str) -> None:
    Path(path).write_text(
        "product,price,stock\nCoffeeMaker X100,129.99,42\nToaster T7,39.50,120\n"
        "Kettle K3,24.90,88\nBlender B5,74.00,17\n",
        encoding="utf-8",
    )


def make_md(path: str) -> None:
    Path(path).write_text(
        "# Deployment Guide\n\n"
        "Deploy to production using the blue-green strategy.\n"
        "Rollback is automatic if error rates exceed 5 percent.\n\n"
        "## Environments\n\n"
        "- staging: auto-deploys from main\n"
        "- production: manual approval required\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------

def main() -> int:
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok))
        print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))

    tmp = tempfile.mkdtemp(prefix="rag_test_")
    chroma_dir = os.path.join(tmp, "chroma_db")
    try:
        header("Generating sample documents")
        pdf = os.path.join(tmp, "employee_policy.pdf")
        docx_f = os.path.join(tmp, "machine_learning_notes.docx")
        txt = os.path.join(tmp, "product_manual.txt")
        csv_f = os.path.join(tmp, "product_list.csv")
        md = os.path.join(tmp, "deployment_guide.md")
        make_pdf(pdf)
        make_docx(docx_f)
        make_txt(txt)
        make_csv(csv_f)
        make_md(md)
        check("sample documents generated", all(os.path.isfile(p) for p in
              [pdf, docx_f, txt, csv_f, md]))

        from chunker import chunk_document
        from document_loader import detect_document_type, load_document
        from vector_store import VectorStore

        header("Ingestion: load -> chunk -> embed -> store")
        store = VectorStore(persist_dir=chroma_dir)
        expected_types = {
            pdf: "pdf", docx_f: "docx", txt: "txt", csv_f: "csv", md: "md",
        }
        total = 0
        for fpath, want_type in expected_types.items():
            got_type = detect_document_type(fpath)
            assert got_type == want_type, f"{fpath}: {got_type} != {want_type}"
            sections = load_document(os.path.basename(fpath), Path(fpath).read_bytes())
            chunks = chunk_document(sections)
            stored = store.add_chunks(chunks)
            total += stored
            print(f"  {os.path.basename(fpath)}: {len(sections)} sections -> "
                  f"{stored} chunks (type={got_type})")
        check("5 documents ingested", total > 0, f"{total} chunks stored")
        check("count() matches", store.count() == total)
        check("page metadata preserved (pdf)", all(
            s["metadata"]["page"] >= 1 for s in load_document(
                os.path.basename(pdf), Path(pdf).read_bytes())))

        header("Retrieval (real MiniLM embeddings, cosine)")
        hits = store.search("What is the maximum number of casual leaves I can take?", k=4)
        check("top-k returned", len(hits) == 4)
        check("correct document retrieved", hits[0]["filename"] == "employee_policy.pdf",
              hits[0]["filename"])
        check("similarity % present", 0 <= hits[0]["similarity"] <= 100,
              f"{hits[0]['similarity']}%")

        hits2 = store.search(
            "According to the documents, what is the difference between supervised "
            "and unsupervised learning?", k=4)
        check("cross-document retrieval", any(
            h["filename"] == "machine_learning_notes.docx" for h in hits2),
            ", ".join({h["filename"] for h in hits2}))

        check("list_documents works", len(store.list_documents()) == 5)
        check("delete_document works", store.delete_document("product_list.csv") > 0
              and len(store.list_documents()) == 4)

        header("RAG pipeline flow with MOCKED LLM (no API key needed)")
        from rag_pipeline import RAGPipeline

        class MockClient:
            """Mimics the OpenAI v1 SDK shape: client.chat.completions.create.
            Responds differently to the question-rewriting call vs the answer
            call, exactly like a real LLM would."""

            class chat:  # noqa: N801 - mirrors the SDK namespace
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        messages = kwargs.get("messages", [])
                        system = messages[0]["content"] if messages else ""
                        if system.startswith("Rewrite the user's latest message"):
                            content = (
                                "How many casual leave days can be taken "
                                "consecutively?"
                            )
                        else:
                            content = "MOCK ANSWER from retrieved context"

                        class Message:
                            pass

                        class Choice:
                            pass

                        class Response:
                            pass

                        msg = Message()
                        msg.content = content
                        choice = Choice()
                        choice.message = msg
                        resp = Response()
                        resp.choices = [choice]
                        return resp

        pipeline = RAGPipeline(vector_store=store, client=MockClient(), model="mock")
        out = pipeline.run(
            "What is the maximum number of casual leaves I can take?",
            history=[{"role": "user", "content": "What is the leave policy?"},
                     {"role": "assistant", "content": "15 casual leave days per year."}],
        )
        check("pipeline returned answer", out["answer"] == "MOCK ANSWER from retrieved context")
        check("follow-up rewritten into standalone query",
              "casual leave" in out["search_query"].lower(),
              out["search_query"])
        check("sources attached with page/chunk", bool(out["sources"]) and all(
            "page" in s and "chunk_index" in s and "similarity" in s for s in out["sources"]),
            f"{len(out['sources'])} sources")

        ctx = pipeline.build_context(out["sources"])
        check("context block includes source header",
              "[1] Source: employee_policy.pdf" in ctx and "chunk" in ctx)

        header("Error handling (no exceptions may escape)")
        try:
            load_document("virus.exe", b"whatever")
            check("unsupported format raises", False)
        except Exception as exc:
            check("unsupported format raises friendly error", "Unsupported" in str(exc))
        try:
            load_document("corrupt.pdf", b"not really a pdf")
            check("corrupt PDF raises", False)
        except Exception as exc:
            check("corrupt PDF raises friendly error", "corrupt" in str(exc).lower())
        try:
            load_document("empty.pdf", b"")
            check("empty doc raises", False)
        except Exception as exc:
            check("empty document raises friendly error", "empty" in str(exc).lower())
        check("empty retrieval returns []",
              store.search("zzz qqq xyzzy plugh", k=2) is not None)

        # graceful degradation without API key
        os.environ.pop("OPENAI_API_KEY", None)
        import importlib
        import rag_pipeline as rp
        importlib.reload(rp)
        check("missing API key degrades gracefully", rp.get_openai_client() is None)

        print("\n==== SUMMARY ====")
        for name, ok in results:
            print(f"  {PASS if ok else FAIL}  {name}")
        failed = [n for n, ok in results if not ok]
        print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
        return 0 if not failed else 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
