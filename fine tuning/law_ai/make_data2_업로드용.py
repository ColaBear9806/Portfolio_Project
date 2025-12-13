from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM,Gemma3ForConditionalGeneration

import torch
import re
# 데이터 출처: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=580
"""
한 번만 로드해서 재사용
"""
model_id = "google/gemma-3-1b-it"

MIN_TOKENS = 50   
MIN_CHARS = 50
QUANT_4_BIT = True

# 양자화 + 모델, 토크나이저 로드

if QUANT_4_BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )


model = Gemma3ForCausalLM.from_pretrained(
model_id,
quantization_config=quant_config,
torch_dtype=torch.bfloat16,
device_map={"": 0},# 모두 gpu에 올리기
attn_implementation="sdpa",

).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)



def summarize_text(sampledata: str, max_new_tokens: int = 170) -> str:
    system_prompt = (
        f"입력된 글을 {max_new_tokens - 20}자 이내로 법률용어 없이 원고의 주장을 요약해줘. "
        "법을 모르는 사람도 이해하기 쉽도록 존댓말로 내용 요약만해줘."
    )#일반인에게 데이터를 입력받는 것을 예상하여, 데이터도 일반인이 입력하는 것처럼 변경

    messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": system_prompt},]},
            {"role": "user","content": [{"type": "text", "text": sampledata},]},
        ], 
    ]
    #토큰화
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        ).to(model.device)

    #모델 결과 출력
    with torch.inference_mode():
          outputs = model.generate(**inputs, 
                                    do_sample=False, 
                                    max_new_tokens=max_new_tokens,)
        
    #입력과 결과가 같이 나오니 입력부분을 제외하고 결과만 출력 
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]    
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return summary


class Datasetup:
    """
    json 한 건을 받아서,
    - 원고 주장 요약
    - 피고 주장 요약
    - 관련 법조항 그대로
    - 결론 요약
    을 합쳐서 하나의 prompt를 만드는 클래스
    """
    QUESTION = (
        "원고 측의 주장과 피고 측의 주장을 살펴보았을때, "
        "내릴 수 있는 판결을 예측하여 관련 법조항과 결론을 작성해줘."
    )
    tokenizer = tokenizer

    token_count: int = 0
    prompt: Optional[str] = None
    include: bool = False

    def __init__(self, data):
        self.data = data
        self.title = data["info"]["caseNm"]
        self.caseNo = data["info"]["caseNo"]
        self.question = Datasetup.QUESTION
        self.parse(data)
        
        #텍스트 정제
    def scrub(self, stuff: str) -> str:
        stuff = str(stuff)
        stuff = stuff.replace("\u3000", " ")
        stuff = re.sub(r'[:\[\]"{}【】\s]+', " ", stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,", ",").replace(",,", ",").replace("'", "").replace("*", "")
        stuff = re.sub(r"[ \t]+", " ", stuff)
        stuff = re.sub(r"\s*\n\s*", "\n", stuff)
        return stuff.strip()

        #기존 데이터는 숫자로 분류한 것을 모델이 알기 쉽도록 글로 변경
    def casefield_num_to_text(self,field):
        text_field = {'1': '민사',
                      '2': '형사',
                      '3': '행정'}
        return text_field[field]

    def detailField_num_to_text(self,field):
        text_field = {'1': '민사',
                      '2': '신청',
                      '3': '가사',
                      '4': '특허',
                      '5': '행정',
                      '6': '형사'}
        return text_field[field]

    def trailField_num_to_text(self,field):
        text_field = {'1': '1심',
                      '2': '2심',}
        return text_field[field]


        #각 데이터 요약 후 본문내용 제작
    def parse(self, json_data):
        contents = ""
        summary = ""
        #사건유형
        caseField = self.casefield_num_to_text(json_data["info"]["caseField"])
        if caseField:
            contents += "사건유형 " + caseField + "\n"

        #세부유형
        detailField = self.detailField_num_to_text(json_data["info"]["detailField"])
        if detailField:
            contents += "세부유형 " + detailField + "\n"
        #심급유형
        trailField = self.trailField_num_to_text(json_data["info"]["trailField"])
        if trailField:
            contents += trailField + "\n"
        else:
            contents += "1심" + "\n"

        # 1 원고 측의 주장 요지
        acusrAssrs = "\n".join(json_data["assrs"]["acusrAssrs"])
        if acusrAssrs:
            contents += "원고 측의 주장 " + summarize_text(acusrAssrs) + "\n"
        else:
            contents += "원고 측의 주장 없음\n"

        # 2 피고 측의 주장 요지
        dedatAssrs = "\n".join(json_data["assrs"]["dedatAssrs"])
        if dedatAssrs:
            contents += "피고 측의 주장 " + summarize_text(dedatAssrs) + "\n"
        else:
            contents += "피고 측의 주장 없음\n"

        bsisFacts = '\n'.join(json_data['facts']['bsisFacts'])
        if bsisFacts:
             contents += "기초사실 " + summarize_text(bsisFacts, 400) + '\n'
        else:
            contents += "기초사실 없음\n"

        # 3 관련 법조항 목록
        relateLaword = "\n".join(json_data["info"]["relateLaword"])
        if relateLaword:
            summary += f"관련 법조항\n{relateLaword}\n"
        else:
            summary += f"관련 법조항 없음\n"

        # 4 결론
        cnclsns = "\n".join(json_data["close"]["cnclsns"])
        if cnclsns:
            summary += "결론 " + cnclsns + "\n"
        else:
            summary += "결론 없음\n"

        # 5 최소 길이 / 토큰 처리
        if len(contents) > MIN_CHARS:
            contents = self.scrub(contents)
            summary = self.scrub(summary)
            self.make_prompt(contents, summary)

        #프롬프트 제작
    def make_prompt(self, contents, summary):
        """
        학습용 prompt 구성:
        [질문] + [원고 주장/피고 주장/관련 법/결론 텍스트]
        """
        self.prompt =   {"messages":
                            [{"role": "system", "content": self.question},
                            {"role": "user", "content": contents},
                            {"role": "assistant", "content": summary}]}

        self. test_prompt = {"messages":
                            [{"role": "system", "content": self.question},
                            {"role": "user", "content": contents},
                            ]}

        chat_text = self.tokenizer.apply_chat_template(
        self.prompt["messages"],
        tokenize=False,              # 문자열로만 받기
        add_generation_prompt=False, # 학습 데이터라서 False
        )

        # 3) 프롬프트를 encode하여 토큰길이 계산
        self.token_count = len(
            self.tokenizer.encode(chat_text, add_special_tokens=False)
        )



    def __repr__(self):
        return f"<{self.title}\n{self.caseNo}>"
