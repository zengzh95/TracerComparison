import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tm
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import BertConfig, BertLMHeadModel
from transformers import XLNetConfig, XLNetLMHeadModel
from transformers import BartConfig, BartModel
from transformers import OPTConfig, OPTModel
from transformers import AlbertConfig, AlbertModel
from transformers import T5EncoderModel,T5Config
from colossalai.nn import CheckpointModule
from registry import non_distributed_component_funcs

HF_BATCH_SIZE = 8
TM_BATCH_SIZE = 64
SEQ_LENGTH = 16

class SimpleNet(nn.Module):
    """
    In this no-leaf module, it has subordinate nn.modules and a nn.Parameter.
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__()
        self.embed = nn.Embedding(32768, 16384)
        self.proj1 = nn.Linear(16384, 8192)
        self.ln1 = nn.LayerNorm(8192)
        self.proj2 = nn.Linear(8192, 16384)
        self.ln2 = nn.LayerNorm(16384)
        self.classifier = nn.Linear(16384, 16384)

    def forward(self, x):
        x = self.embed(x)
        x = self.proj1(x)
        x = self.ln1(x)
        x = self.proj2(x)
        x = self.ln2(x)
        x = self.classifier(x)
        return x


class NetWithRepeatedlyComputedLayers(CheckpointModule):
    """
    This model is to test with layers which go through forward pass multiple times.
    In this model, the fc1 and fc2 call forward twice
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint=checkpoint)
        # self.fc1 = nn.Linear(1024, 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        self.fc1 = nn.LayerNorm(1024)
        self.fc2 = nn.LayerNorm(1024)
        self.fc3 = nn.LayerNorm(1024)
        self.layers = [self.fc1, self.fc2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SubNet(nn.Module):

    def __init__(self, out_features) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, weight):
        return F.linear(x, weight, self.bias)


class NestedNet(CheckpointModule):

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint)
        self.fc1 = nn.Linear(1024, 1024)
        self.sub_fc = SubNet(1024)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sub_fc(x, self.fc1.weight)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class NoLeafModule(CheckpointModule):
    """
    In this no-leaf module, it has subordinate nn.modules and a nn.Parameter.
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint=checkpoint)
        self.proj1 = nn.Linear(1024, 2048)
        self.weight = nn.Parameter(torch.randn(2048, 2048))
        self.proj2 = nn.Linear(2048, 1024)

    def forward(self, x):
        x = self.proj1(x)
        x = F.linear(x, self.weight)
        x = self.proj2(x)
        return x


class GPTLMModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_seq_len=1024,
                 vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(
            GPT2Config(n_embd=hidden_size,
                       n_layer=num_layers,
                       n_head=num_attention_heads,
                       n_positions=max_seq_len,
                       n_ctx=max_seq_len,
                       vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 2048, bias=False)
        self.fc3 = nn.Linear(2048, 768, bias=False)

        self.fc4 = nn.ModuleList()
        for iii in range(5):
            self.fc4.append(nn.Linear(768, 768))

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)

        for mmm in self.fc4:
            out3 = mmm(out3)
        return out3


class BertLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, vocab_size=30522,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = BertLMHeadModel(BertConfig(n_embd=hidden_size, n_layer=num_layers,
                                                n_head=num_attention_heads, max_position_embeddings=768,
                                                vocab_size=vocab_size))

        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class MyBart(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=12, num_attention_heads=16, vocab_size=50265,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = BartModel(BartConfig(d_model=hidden_size, encoder_layers=num_layers, decoder_layers=num_layers,
                                          encoder_attention_heads=num_attention_heads,
                                          decoder_attention_heads=num_attention_heads,
                                          max_position_embeddings=1024,
                                          vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class MyXL(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, vocab_size=32000,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = XLNetLMHeadModel(XLNetConfig(d_model=hidden_size, n_layer=num_layers,
                                                n_head=num_attention_heads, vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class MyOPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = OPTModel(config=OPTConfig(hidden_size=512, num_hidden_layers=6, num_attention_heads=16))

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0]


class MyAlbert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AlbertModel(config=AlbertConfig(embedding_size=128,
                            hidden_size=128,
                            num_hidden_layers=2,
                            num_attention_heads=4,
                            intermediate_size=256))

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                          attention_mask=attention_mask).pooler_output


class MyT5Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = T5EncoderModel(config=T5Config(d_model=512, num_layers=6))

    def forward(self, input_ids):
        # Only return lm_logits
        return self.model(input_ids=input_ids).last_hidden_state


def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def bert_base(checkpoint=False):
    return BertLMModel(hidden_size=768, num_layers=12, num_attention_heads=12, checkpoint=checkpoint)


def xlnet_base(checkpoint=False):
    return MyXL(hidden_size=768, num_layers=12, num_attention_heads=12, checkpoint=checkpoint)


def bart_large(checkpoint=False):
    return MyBart(hidden_size=1024, num_layers=12, num_attention_heads=16, checkpoint=checkpoint)

def opt_model():
    return MyOPT()

def albert_model():
    return MyAlbert()

def t5_encoder_model():
    return MyT5Encoder()


def simple_net(checkpoint=False):
    return SimpleNet(checkpoint=checkpoint)


def albert_data_gen(device="meta"):
    input_ids = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    token_type_ids = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    attention_mask = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    meta_args = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    return meta_args


def opt_data_gen(device="meta"):
    input_ids = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    attention_mask = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
    return kwargs


def t5_data_gen(device="meta"):
    input_ids = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    kwargs = dict(input_ids=input_ids)
    return kwargs


@non_distributed_component_funcs.register(name='bert')
def get_bert_components():
    vocab_size = 30522
    seq_len = 768
    batchSize = 8

    def bert_model_builder(checkpoint=False):
        model = BertLMModel(hidden_size=768, num_layers=12, num_attention_heads=12, vocab_size=vocab_size,
                            checkpoint=checkpoint)
        return model

    def bert_data_gen(device="meta"):
        input_ids = torch.randint(0, vocab_size, (batchSize, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return bert_model_builder, bert_data_gen


@non_distributed_component_funcs.register(name='gpt2')
def get_gpt2_components():
    vocab_size = 50257
    seq_len = 1024
    batchSize = 8

    def gpt2_model_builder(checkpoint=False):
        model = GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, vocab_size=vocab_size,
                           checkpoint=checkpoint)
        return model

    def gpt2_data_gen(device="meta"):
        input_ids = torch.randint(0, vocab_size, (batchSize, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return gpt2_model_builder, gpt2_data_gen


@non_distributed_component_funcs.register(name='albert')
def get_albert_components():
    seq_len = 16
    batchSize = 8

    def albert_model_builder(checkpoint=False):
        model = MyAlbert()
        return model

    def albert_data_gen(device="meta"):
        input_ids = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        token_type_ids = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        attention_mask = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        kwargs = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return kwargs

    return albert_model_builder, albert_data_gen


@non_distributed_component_funcs.register(name='simplenet')
def get_simplenet_components():
    batchSize = 64

    def simplenet_model_builder(checkpoint=False):
        model = SimpleNet(checkpoint=checkpoint)
        return model

    def simplenet_data_gen(device="meta"):
        input_ids = torch.randint(low=0, high=2048, size=(batchSize, 16), device=device)
        kwargs = dict(x=input_ids)
        return kwargs

    return simplenet_model_builder, simplenet_data_gen


@non_distributed_component_funcs.register(name='alexnet')
def get_alexnet_components():
    batchSize = 64

    def alexnet_model_builder(checkpoint=False):
        model = tm.alexnet()
        return model

    def alexnet_data_gen(device="meta"):
        data = torch.rand(int(batchSize), 3, 224, 224, device=device)
        kwargs = dict(x=data)
        return kwargs

    return alexnet_model_builder, alexnet_data_gen


@non_distributed_component_funcs.register(name='vgg16')
def get_vgg16_components():
    batchSize = 64

    def vgg16_model_builder(checkpoint=False):
        model = tm.vgg16()
        return model

    def vgg16_data_gen(device="meta"):
        data = torch.rand(int(batchSize), 3, 224, 224, device=device)
        kwargs = dict(x=data)
        return kwargs

    return vgg16_model_builder, vgg16_data_gen


@non_distributed_component_funcs.register(name='resnet18')
def get_resnet18_components():
    batchSize = 64

    def resnet18_model_builder(checkpoint=False):
        model = tm.resnet18()
        return model

    def resnet18_data_gen(device="meta"):
        data = torch.rand(int(batchSize), 3, 224, 224, device=device)
        kwargs = dict(x=data)
        return kwargs

    return resnet18_model_builder, resnet18_data_gen