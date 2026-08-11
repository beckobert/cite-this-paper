# test_pdf.py

import pymupdf

print("PyMuPDF:", pymupdf.version)

doc = pymupdf.open("./papers/Schran_committee_nn.pdf")

print("Pages:", len(doc))

page = doc[0]

print("\nFirst 1000 characters:")
print(page.get_text()[:1000])
