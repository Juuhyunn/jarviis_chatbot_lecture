# https://jdh5202.tistory.com/850
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import random
import torch
from torch.utils.data import DataLoader, Dataset  # 데이터로더
from gluonnlp.data import SentencepieceTokenizer
#  numpy 설치 후 pip install --upgrade mxnet gluonnlp
from transformers import GPT2Config
# from transformers.configuration_gpt2 import GPT2Config

import gluonnlp
from tqdm import tqdm
import subprocess
import os
import sys
import requests
import hashlib




class MyKoGPT2:
    def __init__(self):
        self.tokenizer = {
            'url':
                'https://kobert.blob.core.windows.net/models/kogpt2/tokenizer/kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
            'fname': 'kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
            'chksum': '818bfa919d'
        }

    def get_tokenizer(self, cachedir='~/kogpt2/'):
        """Get KoGPT2 Tokenizer file path after downloading
        """
        model_info = self.tokenizer
        return self.download(model_info['url'],
                        model_info['fname'],
                        model_info['chksum'],
                        cachedir=cachedir)

    def download(self, url, filename, chksum, cachedir='~/kogpt2/'):
        f_cachedir = os.path.expanduser(cachedir)
        os.makedirs(f_cachedir, exist_ok=True)
        file_path = os.path.join(f_cachedir, filename)
        if os.path.isfile(file_path):
            if hashlib.md5(open(file_path,
                                'rb').read()).hexdigest()[:10] == chksum:
                print('using cached model')
                return file_path
        with open(file_path, 'wb') as f:
            response = requests.get(url, stream=True)
            total = response.headers.get('content-length')

            if total is None:
                f.write(response.content)
            else:
                downloaded = 0
                total = int(total)
                for data in response.iter_content(
                        chunk_size=max(int(total / 1000), 1024 * 1024)):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * downloaded / total)
                    sys.stdout.write('\r[{}{}]'.format('█' * done,
                                                       '.' * (50 - done)))
                    sys.stdout.flush()
        sys.stdout.write('\n')
        assert chksum == hashlib.md5(open(
            file_path, 'rb').read()).hexdigest()[:10], 'corrupted file!'
        return file_path

# ##########################################################################################



    def execute(self):
        print(torch.cuda.device_count())  # GPU deviec count check
        ctx = 'cuda'  # 'cuda' #'cpu' #학습 Device CPU or GPU. colab의 경우 GPU 사용
        cachedir = '~/kogpt2/'  # KoGPT-2 모델 다운로드 경로
        epoch = 200  # 학습 epoch
        save_path = 'drive/My Drive/Colab Notebooks/NarrativeKoGPT2/checkpoint/'
        # use_cuda = True # Colab내 GPU 사용을 위한 값

        pytorch_kogpt2 = {
            'url':
                'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
            'fname': 'pytorch_kogpt2_676e9bcfa7.params',
            'chksum': '676e9bcfa7'
        }
        kogpt2_config = {
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "n_ctx": 1024,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_positions": 1024,
            "vocab_size": 50000
        }
        # download model
        model_info = pytorch_kogpt2
        model_path = self.download(model_info['url'],
                              model_info['fname'],
                              model_info['chksum'],
                              cachedir=cachedir)
        # download vocab
        vocab_info = self.tokenizer
        vocab_path = self.download(vocab_info['url'],
                              vocab_info['fname'],
                              vocab_info['chksum'],
                              cachedir=cachedir)

        # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
        kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
        # model_path로부터 다운로드 받은 내용을 load_state_dict으로 업로드
        kogpt2model.load_state_dict(torch.load(model_path))

        device = torch.device(ctx)
        kogpt2model.to(device)

        # kogpt2model.eval()
        # 추가로 학습하기 위해 .train() 사용
        kogpt2model.train()
        vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                                  mask_token=None,
                                                                  sep_token=None,
                                                                  cls_token=None,
                                                                  unknown_token='<unk>',
                                                                  padding_token='<pad>',
                                                                  bos_token='<s>',
                                                                  eos_token='</s>')
        tok_path = self.get_tokenizer()
        model, vocab = kogpt2model, vocab_b_obj
        sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

        # os.chdir("../")
        data_file_path = 'drive/My Drive/Colab Notebooks/NarrativeKoGPT2/data/backmyo_novel_1/untokenized_bm_data.txt'
        batch_size = 2
        novel_dataset = NovelDataset(data_file_path, vocab, sentencepieceTokenizer)
        novel_data_loader = DataLoader(novel_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        learning_rate = 1e-5
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def get_gpu_memory_map(self):
        """Get the current gpu usage.

        Returns
        -------
        usage: dict
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map

    print('KoGPT-2 Transfer Learning Start')
    epoch = 200
    for epoch in range(epoch):
        count = 0
        for data in novel_data_loader:
            optimizer.zero_grad()
            data = torch.stack(data)  # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.

            data = data.transpose(1, 0)
            data = data.to(ctx)

            outputs = model(data, labels=data)
            loss, logits = outputs[:2]
            loss.backward()
            optimizer.step()
            if count % 10 == 0:
                print('epoch no.{} train no.{}  loss = {}'.format(epoch, count + 1, loss))
                # torch.save(model,save_path+'checkpoint_{}_{}.tar'.format(epoch,count))
                # 추론 및 학습 재개를 위한 일반 체크포인트 저장하기
            if (count > 0 and count % 100 == 0) or (len(data) < batch_size):
                torch.save({
                    'epoch': epoch,
                    'train_no': count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_path + 'narrativeKoGPT2_checkpoint.tar')

            count += 1


class NovelDataset(Dataset):
  """web novel dataset"""

  def __init__(self, file_path,vocab,tokenizer):
    self.file_path = file_path
    self.data =[]
    self.vocab =vocab
    self.tokenizer = tokenizer
    file = open(self.file_path, 'r', encoding='utf-8')

    while True:
      line = file.readline()
      if not line:
        break
      toeknized_line = tokenizer(line[:-1])
      index_of_words = [vocab[vocab.bos_token],] + vocab[toeknized_line]+ [vocab[vocab.eos_token]]

      self.data.append(index_of_words)

    file.close()

if __name__ == "__main__":
    k = MyKoGPT2()
    k.execute()