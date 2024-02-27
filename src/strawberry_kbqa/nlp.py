import nltk
import spacy


class _NLP:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        nltk.download("punkt")

        self.doc = None

    def pos_tag(self, text: str = None):
        self.doc = self.nlp(text) if text else self.doc
        return [(token.text, token.pos_) for token in self.doc]

    def ner(self, text: str = None):
        self.doc = self.nlp(text) if text else self.doc
        return [(ent.text, ent.label_) for ent in self.doc.ents]

    def get_subject_object(self, text: str = None):
        self.doc = self.nlp(text) if text else self.doc
        subs = []
        objs = []
        for token in self.doc:
            if "subj" in token.dep_:
                subs.append(token.text)
            elif "obj" in token.dep_:
                objs.append(token.text)
        return subs, objs


NLP = _NLP()


if __name__ == "__main__":
    # sentence = "Barack Obama was born in Hawaii."
    sentence = "Timmy is from Japan."
    print(NLP.pos_tag(sentence))
    print(NLP.ner(sentence))
    print(NLP.get_subject_object(sentence))
