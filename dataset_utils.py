from datasets import load_dataset

def load_glue_sst2(tokenizer, max_seq_len=128):
    dataset = load_dataset("glue", "sst2")
    prompt_template = "Review: {}\nSentiment: "
    pos_id = tokenizer.encode("positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode("negative", add_special_tokens=False)[0]

    def preprocess(examples):
        inputs = [prompt_template.format(s) for s in examples["sentence"]]
        model_inputs = tokenizer(inputs, max_length=max_seq_len, padding="max_length", truncation=True)
        model_inputs["labels"] = examples["label"]
        return model_inputs

    processed = dataset.map(preprocess, batched=True)
    processed.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return processed, pos_id, neg_id
