import pandas as pd
import spacy
from tqdm import tqdm

# 1. Load the dataset provided by Team Member 1
df = pd.read_csv("../data/contracts_ie_clean.csv")

# Load a multilingual or English spaCy model optimized for web text/entities
# Make sure to run: !python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

print(f"Loaded {len(df)} contracts. Starting entity extraction...")

# 2. Extract Entities (Running a sample process or batch process)
# Note: With 139k rows, processing text can take time. Start by testing on a sample of 1,000 rows!
df_sample = df.sample(1000, random_state=42).copy()

extracted_orgs = []
extracted_locs = []

# FIX: 'cleaned_description' does not exist; correct column is 'description'
for doc in nlp.pipe(df_sample['description'].astype(str), batch_size=50):
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    locs = [ent.text for ent in doc.ents if ent.label_ == "GPE" or ent.label_ == "LOC"]
    
    extracted_orgs.append(orgs)
    extracted_locs.append(locs)

df_sample['extracted_companies'] = extracted_orgs
df_sample['extracted_locations'] = extracted_locs

# 3. Create a Feature Flag (Example: Check for shared structural networks)
# Group by location to see if multiple distinct bidders are operating out of the same place
# We explode the lists to examine them row-by-row
exploded_locs = df_sample.explode('extracted_locations')
location_counts = exploded_locs['extracted_locations'].value_counts()

# Flag locations that appear multiple times across different contracts/bidders
suspicious_locations = location_counts[location_counts > 1].index.tolist()

df_sample['shared_address_flag'] = df_sample['extracted_locations'].apply(
    lambda x: 1 if any(loc in suspicious_locations for loc in x) else 0
)

# 4. Save your engineered features for Team Member 3
# Your output file must map back cleanly using contract_id
ner_features = df_sample[['contract_id', 'shared_address_flag']]
ner_features.to_csv("../data/ner_features.csv", index=False)

print("Task complete! saved ner_features.csv for Team Member 3.")