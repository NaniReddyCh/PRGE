import torch, random, numpy as np, json
from tqdm import tqdm
import torch.nn as nn
from prge_optimizer import PRGEOptimizer
from model_utils import prepare_model_for_prge, count_trainable_parameters
from dataset_utils import load_glue_sst2

def train_model(model, tokenizer, args, device, results_path="results.json"):
    dataset, pos_id, neg_id = load_glue_sst2(tokenizer, args.max_seq_len)
    train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=args.batch_size)

    model = prepare_model_for_prge(model, args.lora_rank, args.lora_alpha).to(device)
    print(f"Trainable params: {count_trainable_parameters(model)}")

    optimizer = PRGEOptimizer(model.parameters(), lr=args.learning_rate, q=args.query_budget, epsilon=args.epsilon)
    criterion = nn.CrossEntropyLoss(reduction="none")
    results = {"steps": [], "accuracy": []}

    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(tqdm(train_loader)):
            input_ids, attn, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            def closure():
                all_losses, seeds = torch.zeros(args.query_budget,2,device=device), []
                for i in range(args.query_budget):
                    seed = random.randint(0, 2**32-1)
                    seeds.append(seed)
                    for k, sign in enumerate([1,-1]):
                        torch.manual_seed(seed)
                        for p in model.parameters():
                            if p.requires_grad:
                                noise = torch.randn_like(p.data)
                                p.data.add_(noise, alpha=sign*args.epsilon)
                        out = model(input_ids=input_ids, attention_mask=attn)
                        logits = out.logits[:,-1,:][:, [neg_id, pos_id]]
                        loss = criterion(logits, labels).mean()
                        all_losses[i,k] = loss
                        torch.manual_seed(seed)
                        for p in model.parameters():
                            if p.requires_grad:
                                noise = torch.randn_like(p.data)
                                p.data.add_(noise, alpha=-sign*args.epsilon)
                return {"losses": all_losses, "seeds": seeds}

            optimizer.step(closure)

            if (step+1) % args.eval_steps == 0:
                acc = evaluate(model, eval_loader, pos_id, neg_id, device)
                print(f"Step {step+1}: Accuracy={acc:.4f}")
                results["steps"].append(step+1)
                results["accuracy"].append(acc)
                with open(results_path,"w") as f: json.dump(results,f)
    return results

def evaluate(model, loader, pos_id, neg_id, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids, attn, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            logits = model(input_ids=ids, attention_mask=attn).logits[:,-1,:][:,[neg_id,pos_id]]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct/total
