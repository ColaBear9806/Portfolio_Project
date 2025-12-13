from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM,Gemma3ForConditionalGeneration

import torch
import re
# 데이터 출처: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=580
"""
한 번만 로드해서 재사용
"""

model_id = "google/gemma-3-1b-it"#텍스트 요약용이므로 더 많은 파라미터의 모델이 필요하지 않음
            #학습용이 아니므로 pt 모델이 아닌 it 모델 사용

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True)

model = Gemma3ForCausalLM.from_pretrained(
    #기존에는 AutoModelForImageTextToText(멀티모달)을 사용했으나 
    #공식에 나온 Gemma3ForCausalLM(텍스트)로 모델 로드하는 방법 사용
model_id,
quantization_config=quant_config,
torch_dtype=torch.bfloat16,
device_map={"": 0},
attn_implementation="sdpa",

).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)



def summarize_text(sampledata: str, max_new_tokens: int = 170) -> str:
    system_prompt = (
        f"입력된 글을 {max_new_tokens - 20}자 이내로 법률용어 없이 원고의 주장을 요약해줘. "
        "법을 모르는 사람도 이해하기 쉽도록 존댓말로 내용 요약만해줘."
    )
    #요약용 프롬프트 생성
    messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": system_prompt},]},
            {"role": "user","content": [{"type": "text", "text": sampledata},]},
        ], 
    ]
    #토큰화
    prompt_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(
        prompt_str,
        return_tensors="pt",
        padding=False,
        truncation=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]    
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return summary


class Summerizer:

    def __init__(self, data):
        self.data = data
        self.summary = self.parse(data)

    def scrub(self, stuff: str) -> str:
        #텍스트 정제
        stuff = str(stuff)
        stuff = stuff.replace("\u3000", " ")
        stuff = re.sub(r'[:\[\]"{}【】\s]+', " ", stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,", ",").replace(",,", ",").replace("'", "").replace("*", "")
        stuff = re.sub(r"[ \t]+", " ", stuff)
        stuff = re.sub(r"\s*\n\s*", "\n", stuff)
        return stuff.strip()

    def parse(self, str_data):
        #data를 받아서 문자열을 만들고,
        #전처리 후 요약본 반환

        str_data = "\n".join(str_data)
        str_data = self.scrub(str_data)
        str_data = summarize_text(str_data)
        str_data = self.scrub(str_data)
        return str_data
