"""
Generates a small synthetic labeled dataset (data/news.csv) for demo purposes.
Replace this with a real dataset (e.g. Kaggle's Fake and Real News Dataset)
for production-quality results. Point train.py at your own CSV with
'text' and 'label' columns (label = FAKE or REAL) to use real data instead.
"""
import csv
import random

random.seed(42)

REAL_TEMPLATES = [
    "The Federal Reserve announced today it will {action} interest rates by {num} basis points.",
    "Scientists at {univ} University published a peer-reviewed study on {topic} in the journal Nature.",
    "The city council voted {num} to {num2} to approve the new public transit budget.",
    "Quarterly earnings for {company} rose {num}% year-over-year, according to the company's official filing.",
    "The World Health Organization released updated guidelines on {topic} based on clinical trial data.",
    "Local officials confirmed that road construction on {street} Street will begin next month.",
    "The national weather service issued a routine forecast update for the {region} region this week.",
    "A new report from the {agency} shows unemployment fell to {num}% last quarter.",
    "The school board approved a revised curriculum for {topic} after a public comment period.",
    "Researchers at {univ} presented preliminary findings on {topic} at a scientific conference.",
]

FAKE_TEMPLATES = [
    "SHOCKING: Doctors HATE this one weird trick that cures {topic} overnight!!!",
    "You won't believe what {celebrity} said about {topic} - the government doesn't want you to know!",
    "BREAKING: Secret documents PROVE {topic} is a total hoax invented by {agency}!",
    "Scientists are TERRIFIED of this new discovery about {topic} that they're hiding from the public.",
    "Anonymous insider reveals {company} is secretly controlling {topic} through mind control chips.",
    "This miracle cure for {topic} was banned by {agency} because it threatens their profits!",
    "Leaked footage shows {celebrity} admitting the whole {topic} story was FAKE all along.",
    "Wake up! {agency} doesn't want you to know the TRUTH about {topic}, share before it's deleted!",
    "URGENT: {celebrity} caught in massive {topic} scandal that mainstream media REFUSES to cover.",
    "New study 'they' don't want you to see proves {topic} was engineered in a secret lab.",
]

fillers = {
    "action": ["raise", "lower", "hold"],
    "num": [2, 3, 5, 8, 12, 25, 50, 61, 74],
    "num2": [1, 2, 3, 4, 6, 9],
    "univ": ["Stanford", "MIT", "Oxford", "Harvard", "Cambridge", "Yale"],
    "topic": ["climate change", "vaccine safety", "the economy", "5G networks", "renewable energy",
              "public health", "artificial intelligence", "election security", "water quality", "nutrition"],
    "company": ["Acme Corp", "Globex", "Initech", "Umbrella Inc", "Stark Industries"],
    "street": ["Main", "Elm", "Oak", "Maple", "5th"],
    "region": ["northeast", "midwest", "southern", "pacific", "central"],
    "agency": ["the CDC", "the FDA", "the UN", "the Fed", "the FBI"],
    "celebrity": ["a famous actor", "a well-known politician", "a popular influencer", "a tech billionaire"],
}


def fill(template):
    out = template
    for key, options in fillers.items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", str(random.choice(options)), 1)
    return out


rows = []
for _ in range(150):
    rows.append((fill(random.choice(REAL_TEMPLATES)), "REAL"))
for _ in range(150):
    rows.append((fill(random.choice(FAKE_TEMPLATES)), "FAKE"))

random.shuffle(rows)

with open("data/news.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to data/news.csv")
