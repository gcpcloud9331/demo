from flask import Flask
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
from healthcheck import HealthCheck
import http.client

app = Flask(__name__)
health = HealthCheck()

def check_health():
    conn = http.client.HTTPConnection("localhost", 8080)
    conn.request("GET", "/")
    res = conn.getresponse()
    if res.status == 200:
        return True, "app healthy"

health.add_check(check_health)

@app.route('/')
def predict():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased',    return_dict = True)
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
    output = model(**input)
    logits = output.logits
    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        return str(new_sentence)
    
if __name__ == '__main__':
    app.add_url_rule("/healthcheck", "healthcheck", view_func=lambda: health.run())
    app.run(host="0.0.0.0", port=8080, debug=True)
