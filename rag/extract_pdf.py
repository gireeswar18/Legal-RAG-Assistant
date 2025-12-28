from pypdf import PdfReader

reader = PdfReader("data/IPC-Book.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

with open("data/book.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("book.txt created")