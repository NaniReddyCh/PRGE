import streamlit as st, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import train_model
from plots import plot_results
import json

st.title("📱 On-Device Fine-tuning (P-RGE + LoRA-FA)")

model_name = st.selectbox("Choose model", ["gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
epochs = st.slider("Epochs", 1, 3, 1)
batch_size = st.selectbox("Batch size", [2,4,8])
query_budget = st.selectbox("Query budget (q)", [2,4,8])
epsilon = st.select_slider("Epsilon (perturbation scale)", [1e-3, 1e-2, 5e-2], value=1e-2)

if st.button("🚀 Start Training"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    class Args: pass
    args = Args()
    args.model_name, args.num_epochs, args.batch_size = model_name, epochs, batch_size
    args.learning_rate, args.eval_steps, args.query_budget, args.epsilon = 5e-4, 50, query_budget, epsilon
    args.lora_rank, args.lora_alpha, args.max_seq_len = 8, 16, 128

    results = train_model(model, tokenizer, args, device)
    
    import datetime
    #save the model
    model.save_pretrained(f"fine_tuned_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    tokenizer.save_pretrained(f"fine_tuned_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # with open("results.json", "w") as f:
    #     json.dump(results, f)

    st.success("Training finished ✅")
    fig = plot_results()
    st.pyplot(fig)
