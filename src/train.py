from datasets.xml_dataset import BertDataset


if __name__ == '__main__':
    i2b2_dataset = BertDataset("../data/token/train", pretrained_tokenizer="bert-base-uncased")
    sentence, ids = i2b2_dataset.tokenizer.decode(i2b2_dataset[3][1].tolist(), i2b2_dataset[3][2].tolist())
    print(' '.join(sentence))
    print(ids)
