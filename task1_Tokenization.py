#step1:(Read Pdf)
from PyPDF2 import PdfReader
reader=PdfReader(r"C:\Users\prade\Downloads\Today.pdf")
full_text=" "
for page in reader.pages:
    text=page.extract_text()
    if text:
        full_text+=text+"\n"
print(len(full_text))

#step2:(Clean Text)
import re
full_text=re.sub(r'\s+',' ',full_text)
full_text=full_text.strip()

#step3:(Load Tokenizer)
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

#step4:(Tokenization)
corpus_tokens=tokenizer.encode(full_text,add_special_token=False)
print("Total Tokens:",len(corpus_tokens))
print("First 100 tokens:")
print(corpus_tokens[:100])        