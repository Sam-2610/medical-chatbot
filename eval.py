"""Run from the medical-chatbot folder after ingest.py:  python eval.py"""
import csv
import time

from rag import answer

# (question, keywords the answer should contain, minimum keywords needed)
# keywords = None means the question is NOT in the documents, so the bot should refuse.
TESTS = [
    ("How are sexually transmitted infections passed from person to person?",
     ["blood", "semen", "vaginal fluid", "cuts"], 3),
    ("Can you get an STI from sharing food, using the same toilet, or insect bites?",
     ["cannot", "can't", "no,", "no.", "not "], 1),
    ("Which STIs can easily pass through oral sex?",
     ["gonorrhea", "herpes", "hpv"], 2),
    ("Which STI can also pass from mother to baby through breastfeeding?",
     ["hiv"], 1),
    ("Why are women's bodies more vulnerable to STIs than men's?",
     ["semen", "thin", "skin"], 2),
    ("How often should people with more than one partner get tested for STIs?",
     ["6", "12"], 2),
    ("What does the book say about hepatitis B?",
     ["vaccine", "expensive", "cure"], 2),
    ("Which lubricants are safe to use with condoms?",
     ["saliva", "water", "oil", "vaseline"], 2),
    ("Which union helped the three women in Ahmedabad organize?",
     ["sewa", "self-employed"], 1),
    ("How long did it take for the women's medicine cooperative to get its permit?",
     ["2 years", "two years"], 1),
    ("What is a power map used for?",
     ["allies", "opponents", "strateg"], 2),
    ("What three strategies for organizing does the book describe?",
     ["consensus", "take power", "make power"], 3),
    ("What is a social audit?",
     ["evaluation", "services", "needs"], 2),
    ("Which health services should be offered at low cost at the community level?",
     ["information", "immuniz", "vaccin", "prenatal", "pregnan", "family planning", "exam"], 3),
    ("What things should anyone helping you with health care provide?",
     ["access", "information", "choice", "safety", "respect", "privacy", "comfort", "follow"], 6),
    ("What is the advice about blood transfusions?",
     ["emergenc", "own blood", "hiv", "hepatitis"], 2),
    ("Which type of anesthetic is safer for an operation, and why?",
     ["local", "safer"], 2),
    # Not in the documents: the bot should say it couldn't find the answer
    ("Who won the 2023 cricket world cup?", None, 0),
    ("What is the adult dose of paracetamol?", None, 0),
    ("How is type 2 diabetes treated?", None, 0),
]

REFUSALS = ["couldn't find", "could not find", "don't know", "not in the documents"]
BUSY = "ai service is busy"


def passed(reply, keywords, need):
    text = reply.lower()
    if keywords is not None and any(r in text for r in REFUSALS):
        return False  # answered but also said "couldn't find" -> contradictory
    if keywords is None:
        return any(r in text for r in REFUSALS)
    return sum(k in text for k in keywords) >= need


rows, ok, skipped, total_time, timed = [], 0, 0, 0.0, 0
for i, (q, kws, need) in enumerate(TESTS, 1):
    t0 = time.time()
    reply, hits = "", []
    for attempt in range(3):
        reply, hits = answer(q)
        if BUSY not in reply.lower():
            break
        print("    rate limited, waiting 20s (please don't press Ctrl+C)...")
        time.sleep(20)
    dt = time.time() - t0

    if BUSY in reply.lower():
        skipped += 1
        print(f"{i:2}. SKIP  (AI service busy)  | {q[:50]}")
        rows.append([i, q, "SKIPPED", round(dt, 1), "-", reply])
        if skipped >= 3:
            print("\nToo many rate-limit errors. Stop here, wait, or switch provider.")
            break
        continue

    result = passed(reply, kws, need)
    ok += result
    total_time += dt
    timed += 1
    top = f"{hits[0]['source']} p{hits[0]['page']}" if hits else "-"
    print(f"{i:2}. {'PASS' if result else 'FAIL'}  {dt:4.1f}s  top: {top}  | {q[:50]}")
    rows.append([i, q, result, round(dt, 1), top, reply])
    time.sleep(4)  # be gentle with free-tier rate limits

if skipped:
    print(f"\nRun incomplete: {skipped} skipped, {len(TESTS) - len(rows)} not run. Don't use this accuracy.")
print(f"Passed: {ok}/{len(TESTS)} = {ok / len(TESTS):.0%}")
if timed:
    print(f"Average response time: {total_time / timed:.1f}s")

with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["#", "question", "pass", "seconds", "top_source", "answer"])
    w.writerows(rows)
print("Saved eval_results.csv")