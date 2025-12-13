from tqdm import tqdm
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset, DatasetDict
import pickle
# 데이터 출처: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=580
"""
Class 개체 생성전 main ipynb에서 import하면 실행해야 될 것들
즉, 한 번만 로드해서 재사용
"""

RUN_NAME = ""
HF_USER = ""
PROJECT_NAME = ""
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "" # or REVISION = None
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"


MODEL_ID = "google/gemma-3-4b-pt"
TOKENIZER_MODEL = "google/gemma-3-4b-it"
HF_USER = ""

if MODEL_ID == "google/gemma-3-1b-pt":
    MODEL_CLASS = AutoModelForCausalLM
else:
    MODEL_CLASS = AutoModelForImageTextToText

if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16


quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )


# 멀티모달 모델은 AutoModelForImageTextToText
base_model = MODEL_CLASS.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    torch_dtype=torch_dtype,
    device_map={"": 0},
    attn_implementation="sdpa",
)

# LoRA 어댑터 모델에 합치기
model = PeftModel.from_pretrained(
    base_model,
    FINETUNED_MODEL,
).eval()

# 토크나이저 (훈련 때와 동일)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)

# embedding 모델은 기존에 vectorstore 구축때 사용했던 모델 사용
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# vectorstore 불러오기
vectorstore = Chroma(
    persist_directory="/vector_db",
    embedding_function=embeddings
)


class predict_law:

    def __init__(self, data):
        self.data = data
        self.prompt = self.make_prompt(data)
        self.prompt = self.rag_add(self.prompt)
        self.answer = self.chat_generate(self.prompt)
        
    #들어온 list를 바탕으로 프롬프트 구현
    def make_prompt(self, user_prompt):
        messages = {'messages': 
                [
                    {'content': user_prompt[3], 'role': 'system'},
                    {'content': "사건유형 " + user_prompt[1] + " 세부유형 " +  user_prompt[2] +  " " +  user_prompt[0] + 
                       " " +  user_prompt[4] + " " + user_prompt[5] + " " + user_prompt[6]
                       ,'role': 'user'},
                    {'content': user_prompt[7],'role': 'assistant'}
                ]
               }
        return messages
    #vectorstore에서 만들어진 프롬프트를 바탕으로 문서 검색 후, 검색된 문서 프롬프트에 추가
    def rag_add(self, prompt):
        retrieved_docs = vectorstore.similarity_search(prompt["messages"][1]["content"], k=3)

        self.rag = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt['messages'][1]['content'] += "\n\n참고자료:\n" + self.rag
        return prompt


    #만들어진 프롬프트를 입력받아 모델 출력 생성
    def chat_generate(self, sample, max_new_tokens: int = 300):
        messages = sample["messages"][:2]  # system + user
    
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # model.generate를 사용하면 입력과 결과가 같이 나오기 때문에 결과만 나오도록 하기
        gen_ids = outputs[0, inputs.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return gen_text

    
