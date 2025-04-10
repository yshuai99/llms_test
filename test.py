from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 检查路径格式是否正确（建议使用原始字符串）
model_dir = r"E:\Yuxin\projects\PAIE-main\llama-2-7b"

# 加载模型和分词器

model = AutoModelForCausalLM.from_pretrained(model_dir, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 测试输入
text = "现在是四月二日，这是一个测试嵌入表示的例子。"
inputs = tokenizer(text, return_tensors="pt")

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)

if outputs.hidden_states is not None:
    print("隐藏层状态形状:", [hs.shape for hs in outputs.hidden_states])
else:
    print("隐藏层状态未返回")

