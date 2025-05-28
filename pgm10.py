# Install spacy if not already installed (uncomment the next line if needed)

# pip install spacy
# python -m spacy download en_core_web_sm

import spacy

# Load a small English language model that contains vocabulary, syntax, and NER
nlp = spacy.load("en_core_web_sm")

# Input sentence (can be from chatbot, search, email, etc.)
text = "Google acquired DeepMind in 2014 for developing artificial intelligence."

# Step 1: Process the text using the NLP pipeline
doc = nlp(text)

# Step 2: Print each word and its Part-of-Speech (POS) tag and dependency relation
print("🔍 Word-Level Semantic and Syntactic Information:\n")
for token in doc:
    print(f"Text: {token.text:15} | POS: {token.pos_:10} | Dependency: {token.dep_:15} | Head: {token.head.text}")

# Step 3: Named Entity Recognition (NER)
print("\n🏷️ Named Entities (Real-world concepts recognized):\n")
for ent in doc.ents:
    print(f"Entity: {ent.text:25} | Label: {ent.label_} | Explanation: {spacy.explain(ent.label_)}")

# Step 4: Print root verb and its subject and object — basic semantic role labeling
print("\n🔗 Semantic Roles (Who did what to whom?):\n")
for token in doc:
    if token.dep_ == "ROOT":  # main verb
        subject = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
        obj = [w for w in token.rights if w.dep_ in ("dobj", "pobj")]
        print(f"Action: {token.text}")
        print(f"Subject(s): {[w.text for w in subject]}")
        print(f"Object(s): {[w.text for w in obj]}")
