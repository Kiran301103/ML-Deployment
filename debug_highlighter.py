"""
Run this once from your project root to diagnose why highlighting isn't working.
It prints:
  1. Raw text extracted from page 1 of the PDF by PyMuPDF
  2. The first evidence chunk from your last search
  3. Whether any phrase from the chunk is found in the PDF text
"""

import re
import fitz  # PyMuPDF

# ── CONFIG — update these two lines ──────────────────────────────────────────
PDF_PATH   = r"Data-Engineering\data\pdfs\level_health_plan_d_table_of_cover__table_of_cover_2025-06-27.pdf"
# Paste one real evidence chunk from your search result here (copy from the
# "📑 Policy chunks" expander in the Streamlit app)
SAMPLE_CHUNK = """range of everyday health expenses. These are consultations that generally take place outside of a hospital setting. You can claim for these through QuickClaims on the Level Health app. Pooling applies to Day to Day medical expenses, meaning you can share visits with other members on the same policy and plan. The benefit for Complementary Therapists covers the number of visit(s) you can claim for in total across all of the listed practitioners, not a number of visits for each type of practitioner. The out-patient excess is the annual amount you must pay out-of-pocket for out-patient treatment before your insurance benefits begin to apply. The claim amount listed for each out-patient benefit is the amount that contributes towards reaching the out-patient excess. This means that if you had an out-patient consultant visit that cost you €200, €50 of that visit would apply towards the out-patient excess amount. 4

Day to Day Medical Expenses GP visits €50 per visit, 6 visits per annum Consultant visits €150 per visit, 2 visits per annum Dentist visits (routine treatment) €50 per visit, 6 visits per annum Physiotherapist visits €50 per visit, 6 visits per annum Psychotherapy and Counselling €50 per visit, 6 visits per annum Complementary Therapists (Acupuncture, Chiropody, Chiropractor, Dietician, Massage Therapist, Nutritionist, Occupational Therapist, Optometrist, Osteopathy, Physical Therapy, Podiatrist, Reflexology, Reiki, Speech and Language Therapist) €50 per visit, 6 visits per annum Health screen €100 x 1 per annum Radiology diagnostic test Up to €20 per test Radiologist fees Up to €25 per test Out-patient Cover (subject to excess) Excess per person €125 Consultant fees €50 per visit Emergency Dental Benefit Up to €250 per claim Pathology diagnostic test Up to €20 per test Pathologist fees Up to €25 per test Overall limit on Day to Day and Out-patient benefits €3,000 Your plan includes cashback for face-to-face visits for a range of everyday health expenses. These are consultations that generally take place outside of a hospital setting. You

Mental Health Assessment Unlimited Expert Medical Opinion Unlimited You can claim back the cost of attending an approved Urgent Care Clinic or Minor Injury Unit, or an Emergency Department in a Private Hospital. Minor injury and urgent care clinics are walk-in medical clinics for treatments of broken bones, sprains, strains and other minor illnesses. A full list of all available Urgent Care Clinics, Minor Injury Units, and Emergency Departments in Private Hospitals can be found here. A full list of all approved scan centres can be found here. Skip the waiting rooms and get immediate medical advice from qualified doctors, midwives and mental health professionals. Whether you need a prescription refill, have a health concern, or need professional reassurance, our doctors are available 24/7, in your pocket. These services can be accesse
"""
# ─────────────────────────────────────────────────────────────────────────────


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


doc = fitz.open(PDF_PATH)
print(f"PDF has {len(doc)} pages\n")

# Print raw extracted text from first 3 pages
for i in range(min(3, len(doc))):
    page = doc[i]
    raw = page.get_text("text")
    print(f"{'='*60}")
    print(f"PAGE {i+1} — raw text ({len(raw)} chars):")
    print(raw[:1000])
    print()

# Now test phrase matching against all pages
print("="*60)
print("CHUNK TEXT (first 300 chars):")
print(SAMPLE_CHUNK[:300])
print()

# Split chunk into candidate phrases
sentences = re.split(r'(?<=[.?!])\s+|\n+', SAMPLE_CHUNK)
phrases = []
for s in sentences:
    s = s.strip()
    if 15 <= len(s) <= 100:
        phrases.append(s)
    elif len(s) > 100:
        for sub in re.split(r'[,;:]\s+', s):
            sub = sub.strip()
            if 15 <= len(sub) <= 100:
                phrases.append(sub)

print(f"Candidate phrases extracted ({len(phrases)}):")
for p in phrases[:10]:
    print(f"  [{len(p)}] {p!r}")

print()
print("SEARCH RESULTS:")
found_any = False
for page_num in range(len(doc)):
    page = doc[page_num]
    for phrase in phrases:
        hits = page.search_for(phrase)
        if hits:
            print(f"  ✅ Page {page_num+1}: FOUND {len(hits)}x — {phrase!r}")
            found_any = True
        else:
            # Try normalized
            page_text = normalize(page.get_text("text"))
            if normalize(phrase) in page_text:
                print(f"  ⚠️  Page {page_num+1}: text match but fitz.search_for missed — {phrase!r}")
                print(f"      → This means whitespace/encoding mismatch in the PDF")
                found_any = True

if not found_any:
    print("  ❌ NO phrases found anywhere in the PDF")
    print()
    print("LIKELY CAUSE: The PDF uses non-standard encoding or is image-based (scanned).")
    print("Check with: pdffonts your_file.pdf")
    print("If text is garbled or empty — the PDF is a scan and needs OCR to highlight.")

doc.close()
