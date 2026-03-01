# Submission Report (LaTeX)

This folder contains the submission-ready report:

- `submission_report.tex`
- `references.bib`

## Build

From this folder:

```bash
pdflatex submission_report.tex
bibtex submission_report
pdflatex submission_report.tex
pdflatex submission_report.tex
```

Output:

- `submission_report.pdf`

## Notes

- The report content is aligned with the current artifacts in `trained_models/`.
- RW validation is documented as auxiliary and non-official.
- `pdflatex` is a system dependency (not a pip package):
  - Windows: `winget install --id MiKTeX.MiKTeX -e`
  - Ubuntu/Debian: `sudo apt-get install texlive-latex-base`
