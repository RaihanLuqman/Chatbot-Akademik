from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import pandas as pd
import re

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

df = pd.read_csv("dataset_chatbot_valid.csv")
chat_history_ids = None

# Normalisasi bahasa kasual
def normalize_input(text):
    text = text.lower()
    slang_dict = {
        "gue": "saya", "gua": "saya", "aku": "saya", "ak": "saya", "sup" : "hi", "hey" : "halo", "hei" : "halo" , 
        "hello" : "halo", "hai" : "halo" , "oi" : "hi" ,"ngajar": "mengajar", "ngasih": "memberikan", "pengen": "ingin",
        "mau": "ingin", "kuliah di sini": "pendaftaran", "masuk kuliah": "pendaftaran",
        "kampus": "universitas", "dosennya siapa": "siapa dosen",
        "formulir": "form", "biaya kuliah": "ukt", "uang kuliah": "ukt",
        "gimana": "bagaimana", "gmn": "bagaimana",  "jurusan": "program studi",  "prodi": "jurusan",
        "ga bisa": "tidak bisa", "gabisa": "tidak bisa", "gak": "tidak",
        "nggak": "tidak", "ga": "tidak", "bgt": "banget", "thanks": "makasih", "terima kasih": "makasih",
        "ty": "makasih", "thx": "makasih", "tq": "makasih", "keren" : "mantap", "gacor" : "mantap"
    }
    filler = [
        "apakah", "bisa", "tolong", "mohon", "kira kira", "dong", "deh", "nih",
        "lho", "ya", "sih", "kan", "oke",
        "admin", "min", "kak", "bang", "pak", "bu", "permisi", "maaf", "wkwk", 
        "hehe", "haha", "btw", "cmn", "cuma", "aja", "cara", "lupa", "banyak"
    ]
    for slang, formal in slang_dict.items():
        text = re.sub(rf"\b{re.escape(slang)}\b", formal, text)
    for f in filler:
        text = re.sub(rf"\b{re.escape(f)}\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Cari pertanyaan yang cocok pada dataset
def search_pertanyaan_response(cleaned_input):
    for _, row in df.iterrows():
        pertanyaan = str(row["pertanyaan"]).lower()
        if pertanyaan in cleaned_input:
            return str(row["jawaban"]).replace("\\n", "\n")
    return None


def generate_response(user_input):
    global chat_history_ids
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if chat_history_ids is not None else input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Chat Function untuk ChatInterface
def chat_fn(message, history):
    cleaned_input = normalize_input(message)
    answer = search_pertanyaan_response(cleaned_input)
    if answer:
        return answer
    else:
        return "Saya masih belum tahu informasi tentang itu."

# Gradio ChatInterface UI
gr.ChatInterface(
    fn=chat_fn,
    title="Chatbot Kampus",
    description="""
    <div style="text-align:center">
        Tanya seputar dosen, kuliah, fasilitas, jurusan, UKM, dll. Tersedia 24 jam 😄
    </div>
    """,
    theme="soft",
    examples=[
        "Saya lupa password",
        "Saya Mau Daftar Dong",
        "Ukm nya ada apa aja?",
        "Jurusannya ada apa aja?",
        "Berapa Biaya Setiap Jurusan?",
    ]
).launch()
