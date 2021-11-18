# https://jdh5202.tistory.com/850
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
class KoClass:
    def __init__(self):
        pass



    def execute(self):
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')

        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        text = '제육볶음을 먹었다'
        input_ids = tokenizer.encode(text)
        gen_ids = model.generate(torch.tensor([input_ids]),
                                 max_length=128,
                                 repetition_penalty=2.0,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id,
                                 bos_token_id=tokenizer.bos_token_id,
                                 use_cache=True)
        generated = tokenizer.decode(gen_ids[0, :].tolist())
        print(generated)


if __name__ == "__main__":
    k = KoClass()
    k.execute()