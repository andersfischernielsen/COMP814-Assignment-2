import glob
import pprint
from segtok.segmenter import split_single
from flair.models import SequenceTagger
from flair.data import Sentence, Span


def get_flair_taggers():
    frame_tagger = SequenceTagger.load('frame-fast')
    ner_tagger = SequenceTagger.load('ner-fast')
    pos_tagger = SequenceTagger.load('pos')
    return ner_tagger, frame_tagger, pos_tagger


def get_organisations(sentence: Sentence) -> Span:
    org_tags = list(filter(lambda span: "ORG" in span.tag,
                           sentence.get_spans('ner')))
    return org_tags


def get_reason_for_appearance(organisation: Span, sentence: Sentence) -> str:
    org_end = organisation.end_pos
    frame_tags = sentence.get_spans('frame')
    pos_tags = list(filter(lambda span: "VBD" in span.tag,
                           sentence.get_spans('pos')))
    frame_tags_after_org = list(
        filter(lambda span: span.start_pos > org_end, frame_tags))
    pos_tags_after_org = list(
        filter(lambda span: span.start_pos > org_end, pos_tags))
    if not frame_tags_after_org and not pos_tags_after_org:
        return None

    first_after_org = frame_tags_after_org[0] if frame_tags_after_org else pos_tags_after_org[0]
    original = sentence.to_original_text()
    end_of_reason = original.find(',', first_after_org.start_pos)
    if not end_of_reason:
        end_of_reason = original.find('.', first_after_org.start_pos)
    reason = original[first_after_org.start_pos:end_of_reason]
    return reason


def add_to_organisation(name, reason, counts, reasons):
    if name in reasons and reason:
        reasons[name].append(reason)
        counts[name] = counts[name] + 1
    elif reason:
        reasons[name] = [reason]
        counts[name] = 1
    else:
        reasons[name] = []
        counts[name] = 1


def pretty_print(reasons, counts):
    pp = pprint.PrettyPrinter()
    pp.pprint(reasons)
    pp.pprint(counts)


def find_organisations(folder: str):
    ner_tagger, frame_tagger, pos_tagger = get_flair_taggers()
    org_reasons = {}
    org_counts = {}
    for path in glob.glob(f'{folder}/*.txt'):
        file = open(path, "r")
        lines = split_single(file.read())
        for line in lines:
            sentence = Sentence(line)
            ner_tagger.predict(sentence)
            frame_tagger.predict(sentence)
            pos_tagger.predict(sentence)
            organisations = get_organisations(sentence)
            if not organisations:
                continue
            for organisation in organisations:
                name = organisation.text
                reason = get_reason_for_appearance(organisation, sentence)
                add_to_organisation(name, reason, org_counts, org_reasons)
        print(f"Finished processing {path}")

    return org_reasons, org_counts


reasons, counts = find_organisations("CCAT")
pretty_print(reasons, counts)
