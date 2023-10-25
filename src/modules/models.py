import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def default(model_path: str, device_map):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map=device_map)
    return model, tokenizer
    
def get_model(params):
    
    func_dict = {
        'marianmt': default,
        't5': default,
        'flan': default,
    }

    return func_dict[params['family']](params['model_path'], params['device_map'])
    