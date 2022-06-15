from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config
import torch
import torch.nn as nn

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, args):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(args.drop_path_rate)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CodeSimModel(nn.Module):
    def __init__(self, args):
        super(CodeSimModel,self).__init__()
        self.config = T5Config.from_pretrained(args.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        
        self.encoder = T5ForConditionalGeneration.from_pretrained(args.pretrained_model)
        self.classifier = RobertaClassificationHead(self.config, args)
        
    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, source_ids):
        source_ids = source_ids.view(-1, 512)
        vec = self.get_t5_vec(source_ids)
        logits = self.classifier(vec)

        return logits