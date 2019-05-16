#!/usr/bin/env python3

import traceback
import glob
import pprint
import sys
from syntok.segmenter import process
from flair.models import SequenceTagger
from flair.data import Sentence, Span


def find_organisations_reasons(folder: str, org_reasons: dict, org_counts: dict):
    ner_tagger, frame_tagger, pos_tagger = get_flair_taggers()
    files_processed = 1
    files = glob.glob(f'{folder}/*.txt')
    print(f"Processing {len(files)} files in '{folder}'.")
    for path in files:
        print(f"[{files_processed}/{len(files)}] Processing {path}...")
        file = open(path, "r")
        paragraphs = process(file.read())
        for sentences_tokenized in paragraphs:
            for tokens in sentences_tokenized:
                sentence = ""
                for token in tokens:
                    sentence += f"{token.spacing}{token.value}"
                sentence = Sentence(sentence.strip())
                ner_tagger.predict(sentence)
                frame_tagger.predict(sentence)
                pos_tagger.predict(sentence)
                organisations = get_organisations(sentence)
                if not organisations:
                    continue

                for first in organisations[:1]:
                    name = clean_organization(first.text)
                    reason = get_reason_for_appearance(first, sentence)
                    add_to_organisation(name, reason, org_counts, org_reasons)

                for remaining in organisations[1:]:
                    name = clean_organization(remaining.text)
                    add_to_organisation(name, None, org_counts, org_reasons)

        files_processed += 1

    print(f"Finished processing {len(files)} files.")


def get_flair_taggers() -> (SequenceTagger, SequenceTagger, SequenceTagger):
    print("Loading flair models...")
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
    reason = original[first_after_org.start_pos:]
    return reason


def clean_organization(full_text: str) -> str:
    cleaned = full_text.strip().lower().replace("--", "").replace("\"", "") \
        .replace("'s", "").replace("'", "").replace("(", "").replace(")", "")
    if "," in cleaned:
        cleaned = cleaned.split(",")[0]
    if "." in cleaned:
        cleaned = cleaned.split(".")[0]
    cleaned = cleaned.replace(",", "")
    split = cleaned.split(" ")
    cleaned = split[0].capitalize()
    for s in split[1:]:
        cleaned = cleaned if len(s) < 4 else f"{cleaned} {s.capitalize()}"
    return cleaned


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


def pretty_print(*args):
    pp = pprint.PrettyPrinter()
    for to_print in args:
        pp.pprint(to_print)


def find_top_five(counts, reasons):
    c_top_five = dict(
        sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5])
    r_top_five = dict((item[0], reasons[item[0]][::-1])
                      for item in c_top_five.items())
    return r_top_five, c_top_five


def main():
    reasons = {}
    counts = {}
    try:
        if len(sys.argv) < 2:
            print()
            sys.exit(
                "Please supply a path for text processing (e.g. 'CCAT') as an argument for this script.")

        find_organisations_reasons(sys.argv[1], reasons, counts)
        top_five_reasons, _ = find_top_five(counts, reasons)
        pretty_print(top_five_reasons)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        print("Results:")
        top_five_reasons, _ = find_top_five(counts, reasons)
        pretty_print(top_five_reasons)
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
