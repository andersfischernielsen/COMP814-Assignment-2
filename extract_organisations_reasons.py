import glob
from flair.models import SequenceTagger
from flair.data import Sentence, Span


def get_flair_taggers():
    frame_tagger = SequenceTagger.load('frame-fast')
    ner_tagger = SequenceTagger.load('ner-fast')
    return ner_tagger, frame_tagger


def get_first_organisation(sentence: Sentence) -> Span:
    org_tags = list(filter(lambda span: "ORG" in span.tag,
                           sentence.get_spans('ner')))
    if org_tags:
        return org_tags[0]
    return None


def get_reason_for_appearance(organisation: Span, sentence: Sentence) -> str:
    org_end = organisation.end_pos
    frame_tags = sentence.get_spans('frame')
    after_org = list(filter(lambda span: span.start_pos > org_end, frame_tags))
    if not after_org:
        return None

    first_after_org = after_org[0]
    original = sentence.to_original_text()
    end_of_reason = original.find(',', first_after_org.start_pos)
    if not end_of_reason:
        end_of_reason = original.find('.', first_after_org.start_pos)
    reason = original[first_after_org.start_pos:end_of_reason]
    return reason


def find_organisations(folder: str):
    ner_tagger, frame_tagger = get_flair_taggers()
    organisations = {}
    for path in glob.glob(f'{folder}/*.txt'):
        file = open(path, "r")
        print(path)
        lines = file.readlines()
        print("\nOriginal text:")
        print(" ".join(lines))
        for line in lines:
            sentence = Sentence(line)
            ner_tagger.predict(sentence)
            frame_tagger.predict(sentence)
            organisation = get_first_organisation(sentence)
            if not organisation:
                continue

            name = organisation.text
            reason = get_reason_for_appearance(organisation, sentence)
            print(f"{name}\n{reason}")
            if name in organisations and reason:
                organisations[name].append(reason)
            elif reason:
                organisations[name] = [reason]
            else:
                organisations[name] = []

    return organisations


orgs = find_organisations("CCAT")
print(orgs.keys)
