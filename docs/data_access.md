# Data access, privacy, and governance

## The dataset is NOT in this repository

No participant data of any kind is distributed with this repository:
no raw ECoG, no participant tables, no subject identifiers, no dates
of service, no imaging, no per-subject outputs. The pipeline works on
a deterministic synthetic cohort (`curve_icon_dbso.synthetic`) so the
code can be verified without any data access.

## Source of the real data

The real (restricted) data used in the historical pilot came from the
USC **DABI** initiative (Data Archive for the BRAIN Initiative) and
was accessed through the Informatics and Computing in Neuroscience
(**ICON**) Lab at the University of Southern California.

| Item | Status |
|---|---|
| Exact DABI dataset name | **[TO BE SUPPLIED by the repository owner]** |
| DABI accession / project ID | **[TO BE SUPPLIED]** |
| Contributing investigators | **[TO BE SUPPLIED]** |
| Required dataset citation | **[TO BE SUPPLIED]** |
| Data Use Agreement (DUA) reference | **[TO BE SUPPLIED]** |

The repository owner must supply these details and the required
citation before this project is shared beyond a private audience;
until then the dataset must be treated as fully restricted.

## If you are an authorized user

1. Obtain your own authorization for the dataset through DABI/USC and
   comply with the applicable Data Use Agreement. A code repository
   license — even where one exists — grants **no** rights to the
   underlying clinical data.
2. Store the data locally, outside this repository (the `.gitignore`
   blocks common data file types, but do not rely on that alone).
3. Run the extraction step against your local copy, for example:

   ```bash
   python scripts/run_hfo_extraction.py \
     --input-dir /your/local/path/all_subs_preprocessed_data \
     --output-csv /your/local/outputs/hfo_channels.csv \
     --aggregated-csv /your/local/outputs/hfo_subjects.csv \
     --band-low 80 --band-high 500
   ```

4. Keep any derived, participant-level outputs outside the repository.
   Only synthetic fixtures and sufficiently aggregated, approved
   summary numbers may be committed — with provenance stated.

## Privacy rules observed in this repository

- Subject codes must not appear in committed files. (Historical
  subject codes appear only inside the archived legacy script as
  evidence of the placeholder-data fallback; no clinical values are
  attached to them.)
- No dates, records, imaging metadata, or intermediate datasets are
  tracked.
- `data/` and `outputs/` are gitignored; `*.csv`, `*.mat`, and raw
  imaging formats are blocked repo-wide.
