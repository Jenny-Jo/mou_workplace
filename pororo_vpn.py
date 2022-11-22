import os
from collections import defaultdict

import nltk
from fairseq import hub_utils

from pororo.models.brainbert.CharBrainRoBERTa import CharBrainRobertaHubInterface, CharBrainRobertaModel
from pororo.tasks import PororoTokenizationFactory
from pororo.tasks.named_entity_recognition import PororoBertCharNer
from pororo.tasks.tokenization import PororoSentTokenizer
from pororo.tasks.utils.base import TaskConfig


class CustomCharBrainRobertaModel(CharBrainRobertaModel):
    @classmethod
    def load_model(cls, ckpt_dir: str, **kwargs):
        x = hub_utils.from_pretrained(ckpt_dir,"model.pt",ckpt_dir,load_checkpoint_heads=True,**kwargs,)
        return CharBrainRobertaHubInterface(x["args"],x["task"],x["models"][0],)

def sent_tokenize(text, language="english"):
    tokenizer = nltk.data.load(f"{nltk_dir}/{language}.pickle")
    return tokenizer.tokenize(text)

class CustomPororoTokenizationFactory(PororoTokenizationFactory):
    def load(self, device: str):
        if "sent" in self.config.n_model:
            return PororoSentTokenizer(sent_tokenize, self.config)
        else:
            super().load(device)

if __name__=='__main__':
    # @todo https://twg.kakaocdn.net/pororo/ko/models/bert/charbert.base.ko.ner.zip 다운받아서 적당한 곳에 "풀어서" 저장해야 함.
    # @todo https://twg.kakaocdn.net/pororo/ko/models/misc/wiki.ko.items 다운받아서 적당한 곳에 저장해야 함.
    # @todo https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip 다운받아서 적당한 곳에 "풀어서" 저장해야 함.

    # charbert.base.ko.ner.zip 압축해제한 폴더.
    ckpt_dir=os.path.abspath('model/ner/charbert.base.ko.ner')

    # wiki.ko.items 경로
    dict_path=os.path.abspath('model/ner/wiki.ko.items')

    # punkt.zip 압축해제한 폴더
    nltk_dir=os.path.abspath('model/ner/punkt')

    lang='ko'
    device='cuda'

    sent_tokenizer = CustomPororoTokenizationFactory(
        task="tokenization",
        model="sent_ko",
        lang=lang,
    ).load(device)

    f_wsd_dict = open(dict_path,"r")
    wsd_dict = defaultdict(dict)
    for line in f_wsd_dict.readlines():
        origin, target, word = line.strip().split("\t")
        wsd_dict[origin][word] = target

    roberta_model=CustomCharBrainRobertaModel.load_model(ckpt_dir=ckpt_dir).eval().to(device)

    ner_model=PororoBertCharNer(
        roberta_model,
        sent_tokenizer,
        wsd_dict,
        device,
        config=TaskConfig(task='ner', lang=lang, n_model='charbert.base.ko.ner'),
    )
    print(ner_model("손흥민은 28세의 183 센티미터, 77 킬로그램이며, 현재 주급은 약 3억 원이다."))
