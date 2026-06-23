import random
import numpy as np
from .dataset_t2m import Text2MotionDataset


class Text2MotionDatasetEval(Text2MotionDataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        w_vectorizer,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        max_text_len=20,
        deterministic=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        self.w_vectorizer = w_vectorizer
        self.max_text_len = max_text_len
        self.deterministic = deterministic


    def __getitem__(self, item):
        # Get text data
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # Randomly select a caption unless deterministic test mode is enabled.
        text_data = text_list[0] if self.deterministic else random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        # HumanML3D stores evaluator tokens as "word/POS", while some custom
        # datasets only store plain normalized words. The text-motion evaluator
        # still needs a POS slot for the one-hot vector, so fall back to OTHER
        # instead of failing during sanity-check validation.
        tokens = [
            token if "/" in token and len(token.rsplit("/", 1)[0]) > 0
            and len(token.rsplit("/", 1)[1]) > 0 else f"{token}/OTHER"
            for token in tokens
        ]

        # Text
        max_text_len = self.max_text_len
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        # Random crop unless deterministic test mode is enabled.
        if self.deterministic:
            coin2 = "single"
        elif self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        m_length = min(m_length, self.max_motion_length)

        if self.deterministic:
            idx = max(0, (len(motion) - m_length) // 2)
        else:
            idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        # Z Normalization
        motion = (motion - self.mean) / self.std

        return caption, motion, m_length, word_embeddings, pos_one_hots, sent_len, "_".join(
            tokens), all_captions
