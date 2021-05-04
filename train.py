from transformers import (TextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer,
                          TrainingArguments,
                          AutoTokenizer,
                          AutoModelForCausalLM)


class TrainerGPT2():

    def __init__(self, train='train.txt',
                 dev='dev.txt',
                 model_name='aubmindlab/aragpt2-base',
                 model_out='model/gpt2_tn',
                 block_size=128,
                 auto=False
                 ):

        self.train = train
        self.dev = dev
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if auto:
            self.load_dataset(block_size)
            self.init_train_args(out_dir=model_out)
            self.do_train()
            self.do_eval()

    def load_dataset(self, block_size=128):

        self.train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=self.train,
            block_size=block_size
        )
        self.dev_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=self.dev,
            block_size=block_size
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

    def init_train_args(self, out_dir, epochs=3,
                        batch_size_train=16, batch_size_dev=32,
                        eval_steps=400, save_steps=800, warmup_steps=500,
                        predict_loss_only=True
                        ):
        self.args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_eval_batch_size=batch_size_dev,
            per_device_train_batch_size=batch_size_train,
            eval_steps=eval_steps,
            save_steps=save_steps,
            warmup_steps=warmup_steps,
            prediction_loss_only=predict_loss_only
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset
        )

    def do_train(self):
        self.trainer.train()
        self.trainer.save_model()

    def do_eval(self):
        self.trainer.evaluate()
