import fitz 
from tqdm.auto import tqdm 
from spacy.lang.en import English 
import re
import pandas as pd
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

import numpy as np
import torch

nlp = English()

nlp.add_pipe("sentencizer")

from huggingface_hub import login

login(token="hf_")






import textwrap

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)




from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)


if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
  attn_implementation = "flash_attention_2"
else:
  attn_implementation = "sdpa"
print(f"[INFO] Using attention implementation: {attn_implementation}")


use_quantization_config = False


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
                                                 torch_dtype=torch.float16, # datatype to use, we want float16
                                                 quantization_config=quantization_config if use_quantization_config else None,
                                                 low_cpu_mem_usage=True, 
                                                 attn_implementation=attn_implementation) # which attention version to use

if not use_quantization_config: # quantization takes care of device setting automatically, so if it's not used, send model to GPU
    llm_model.to("cuda")


class TextProcess:

    def __init__(self, pdf, num_of_chunks):
        self.pdf =  pdf
        self.num_of_chunks = 9
        self.chunks_list = []
        #self.chunk_dict = {}

    def text_cleaner(self, text):
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text

    def process_pdf(self):

        document = fitz.open(self.pdf)
        prim_list = []
        for page_number, page in tqdm(enumerate(document)):
            text = page.get_text()
            text = self.text_cleaner(text)
            prim_list.append({"page_number" : page_number,
                       "num_chars" : len(text),
                        "num_tokens" : len(text) / 4,
                       "num_of words" : len(text.split(" ")),
                       "num_of_sents" : len(text.split(". ")),
                        "text" : text})


        return prim_list

    def create_sentences(self):

        prim_dict =  self.process_pdf()

        for item in tqdm(prim_dict):
            item["sentences"] = list(nlp(item["text"]).sents)

            # Make sure all sentences are strings
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            item["page_sentence_count_spacy"] = len(item["sentences"])



        return prim_dict



    def chunk_maker(self, input_list):
        return [input_list[i:i + self.num_of_chunks] for i in range(0, len(input_list), self.num_of_chunks)]

    def sentence_chunk_maker(self):

        page_dict = self.create_sentences()
        for item in tqdm(page_dict):
            item["sentence_chunks"] = self.chunk_maker(input_list=item["sentences"])
            item["num_chunks"] =  len(item["sentence_chunks"])




        return page_dict

    def chunk_dict_maker(self):

        page_dict = self.sentence_chunk_maker()
        for item in tqdm(page_dict):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  
                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

                self.chunks_list.append(chunk_dict)


        return self.chunks_list










embedding_model = SentenceTransformer(model_name_or_path="sentence-transformers/all-mpnet-base-v2",
                                      device="cuda")
class EmbeddedRAG:
    def __init__(self, list_to_embed, query):
        self.list_to_embed = list_to_embed
        self.model = embedding_model
        self.query = query

    def create_embeddings(self):
        embedding_model.to(
            "cuda")  # requires a GPU installed,

        for item in tqdm(self.list_to_embed):
            item["embedding"] = embedding_model.encode(item["sentence_chunk"])

        return self.list_to_embed


    def save_embeddings(self):
        embedded_framework = self.create_embeddings()
        embedded_framework_df = pd.DataFrame(embedded_framework)
        #embedded_framework_path = "E:\\Experiment\\pythonProject\\rag\\simple-local-rag\\assignment\\processed_files\\embedded_framework.csv"
        #embedded_framework_df.to_csv(embedded_framework_path, index=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Import texts and embedding df
        #embedded_framework_df = pd.read_csv("embedded_framework.csv")

        #embedded_framework_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
            #lambda x: np.fromstring(x.strip("[]"), sep=" "))
        embedded_chunks  = embedded_framework_df.to_dict(orient="records")

        #pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

        # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
        embeddings = torch.tensor(np.array(embedded_framework_df["embedding"].tolist()), dtype=torch.float32).to(
            device)

        return embedded_chunks, embeddings

    def retrieve_relevant_context(self, query: str, embeddings: torch.tensor,
                                        n_resources_to_return: int = 1):


            query_embedding = self.model.encode(query,
                                           convert_to_tensor=True)

            # Get dot product scores on embeddings
            dot_scores = util.dot_score(query_embedding, embeddings)[0]


            scores, indices = torch.topk(input=dot_scores,
                                         k=n_resources_to_return)

            return scores, indices

    def prompt_formatter(self, query: str, context_items):
        """
        Augments query with text-based context from context_items.
        """
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

       
        base_prompt = """Based on the following context items, please answer the query. Your name is miniRAG. 
        Give yourself room to think by extracting relevant passages from the context before answering the query.
        Don't return the thinking, only return the answer.
        Make sure your answers are as explanatory as possible. 
        Use the following examples as reference for the ideal answer style.
        \nExample2:
        Query: Who are you?
        Answer: I am miniRAG, a helpful and efficient assistant designed to provide answers based on context items. 
        \nExample 1:
        Query: What are the fat-soluble vitamins?
        Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
        \nExample 2:
        Query: What are the causes of type 2 diabetes?
        Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
        \nExample 3:
        Query: What is the importance of hydration for physical performance?
        Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
        \nExample 4:
        Query: Explain the technical features of the BMW 310 RR
        Answer: The BMW G310 RR is equipped with a range of modern technological features that enhance both rider safety and convenience. It boasts a fully digital instrument cluster that displays essential information such as speed, engine temperature, trip data, and fuel efficiency in a clear and easy-to-read format. The instrument cluster is well-lit, ensuring good visibility even in low-light conditions.
        \nNow use the following context items to answer the user query:
        {context}
        \nRelevant passages: <extract relevant passages from the context here>
        User query: {query}
        Answer:"""

        base_prompt = base_prompt.format(context=context, query=query)

        dialogue_template = [
            {"role": "user",
             "content": base_prompt}
        ]

        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return prompt

    def ask(self, temperature=0.7,
            max_new_tokens=512,
            format_answer_text=True,
            return_answer_only=True):


        embedded_chunks, embeddings = self.save_embeddings()
        scores, indices = self.retrieve_relevant_context(self.query,embeddings )


        context_items = [embedded_chunks[i] for i in indices]

        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu()  # return score back to CPU


        prompt = self.prompt_formatter(query= self.query,
                                  context_items=context_items)

        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = llm_model.generate(**input_ids,
                                     temperature=temperature,
                                     do_sample=True,
                                     max_new_tokens=max_new_tokens)

        output_text = tokenizer.decode(outputs[0])

        if format_answer_text:
            output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace(
                "Sure, here is the answer to the user query:\n\n", "").replace("<s>", "").replace("</s>", "")

        if return_answer_only:
            return output_text

        return output_text