# Unlikelihood Training (with Harry Potter)

### Description

Using unlikelihood training ([Neural Text Generation with Unlikelihood Training](https://arxiv.org/pdf/1908.04319.pdf)) to fine-tune GPT-2 language model to generate Harry Potter story.

### Dependencies
1. Install `pytorch` and `transformers`: `pip install torch transformers`.
2. Download pretrained GPT-2 model (including tokenizer) and save it to a folder name `GPT-2` in the same level as this codebase.

### Usage
```
python train.py --pretrained_path=./GPT-2 --file=harrypotter.txt --seq_len=200 --epochs=10 --batch_size=16 --learning_rate=0.001 --completion_length=100 --prefix_length=50 --training_type=unlikelihood --save_path=./models
```

### Example of continuation

Prompt: 
```
Harry went to Hogwarts
```

Continuation (greedy decoding with MLE training): 
```
Harry went to Hogwarts, and he was not alone.
"I think he was," said Harry, "and he was not alone. He was in the castle, and he was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone. He was not alone
```

Continuation (greedy decoding with unlikelihood training): 
```
Harry went to Hogwarts.  He had been in the hospital wing for a week, and had been spotted by the Ministry's security staff.  He had been given a new wand by Professor Sprout, who had been very impressed with Harry's ability to read the Ministry's mind.  He had also been given the name of the Patronus Charm, which means "the most powerful wizard in the world."
Harry had been in the Triwizard Tournament, and the last thing he wanted to do was be expelled from Hogwarts.  Dumbledore had given him the Order of the Phoenix, and he will be missed by all.
Ron Weasley, Professor McGonagall, and Sirius Black
```