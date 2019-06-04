#!/usr/bin/env python3

import traceback
import glob
import pprint
import sys
import json
import os
from syntok.segmenter import process
from flair.models import SequenceTagger
from flair.data import Sentence, Span


def find_organisations_reasons(folder: str):
    """ Go through files in the given folder, extract organisation names
        and their reason for appearance in file. """
    org_reasons, org_counts = {}, {}
    try:
        # Get flair models.
        ner_tagger, frame_tagger, pos_tagger = get_flair_taggers()
        # Fetch results from cache, if present.
        files_processed, org_reasons, org_counts = check_cache()
        file_count = 1 if len(files_processed) == 0 \
            else len(files_processed) + 1
        # Find files to process from path.
        files = glob.glob(f"{folder}/*.txt")
        print(f"Processing {len(files)} files in '{folder}'.")
        # Remove previously processed file names.
        to_process = [f for f in files if f not in files_processed]
        for path in to_process:
            print(f"[{file_count}/{len(files)}] Processing {path}...")
            file = open(path, "r")
            # Go through paragraphs sentence by sentence and extract information.
            paragraphs = process(file.read())
            for sentences_tokenized in paragraphs:
                for tokens in sentences_tokenized:
                    sentence = ""
                    for token in tokens:
                        sentence += f"{token.spacing}{token.value}"
                    sentence = Sentence(sentence.strip())
                    # Add NER, POS and Semantic Frame Detection tags to sentence.
                    ner_tagger.predict(sentence)
                    frame_tagger.predict(sentence)
                    pos_tagger.predict(sentence)
                    # Extract all organisations.
                    organisations = get_organisations(sentence)
                    if not organisations:
                        continue

                    # Find the first organisation occurence and its reason for appearance.
                    for first in organisations[:1]:
                        name = clean_organization(first.text)
                        reason = get_reason_for_appearance(first, sentence)
                        add_to_organisation(
                            name, reason, org_counts, org_reasons)

                    # Count remaining organisations, but don't find its reason for appearance,
                    # since the other organisations following the first one don't have meaningful reasons,
                    # leading to broken sentences.
                    for remaining in organisations[1:]:
                        name = clean_organization(remaining.text)
                        add_to_organisation(
                            name, None, org_counts, org_reasons)

            files_processed.append(path)
            # Store in cache after processing.
            dump_to_cache(files_processed, org_reasons, org_counts)
            file_count += 1

        org_reasons.pop('I', None), org_counts.pop('I', None)
        org_reasons.pop('We', None), org_counts.pop('We', None)
        print(f"\nFinished processing {file_count} files.")
        return org_reasons, org_counts
    except:
        # Handle early exit by user (CTRL+C).
        print("\n\nExiting...")
        print(f"Finished processing {file_count} files.")
        return org_reasons, org_counts


def check_cache():
    """ Fetch previously processed results, if present. """
    try:
        processed_files = json.load(open("cache/files.json", "r"))
        org_reasons = json.load(open("cache/org_reasons.json", "r"))
        org_counts = json.load(open("cache/org_counts.json", "r"))
        return processed_files, org_reasons, org_counts
    except:
        return [], {}, {}


def dump_to_cache(processed_files, org_reasons, org_counts):
    """ Dump processed results to cache. """
    try:
        if not os.path.exists("cache"):
            os.makedirs("cache")
        json.dump(processed_files, open("cache/files.json", "w"))
        json.dump(org_reasons, open("cache/org_reasons.json", "w"))
        json.dump(org_counts, open("cache/org_counts.json", "w"))
    except:
        return


def get_flair_taggers():
    """ Get the Flair tagger and load their respective models."""
    print("Loading flair models...")
    frame_tagger = SequenceTagger.load("frame-fast")
    ner_tagger = SequenceTagger.load("ner-fast")
    pos_tagger = SequenceTagger.load("pos")
    return ner_tagger, frame_tagger, pos_tagger


def get_organisations(sentence: Sentence):
    """ Extract 'ORG' NER tags in a sentence """
    org_tags = list(filter(lambda span: "ORG" in span.tag,
                           sentence.get_spans("ner")))
    return org_tags


def get_reason_for_appearance(organisation: Span, sentence: Sentence):
    """ Extract the reason for the appearance of an 'ORG' NER tag in a sentence. """
    # Find ORG placement in sentence.
    org_end = organisation.end_pos
    frame_tags = sentence.get_spans("frame")
    # Extract frame and POS tags after organisation occurence.
    pos_tags = list(filter(lambda span: "VBD" in span.tag,
                           sentence.get_spans("pos")))
    frame_tags_after_org = list(
        filter(lambda span: span.start_pos > org_end, frame_tags)
    )
    pos_tags_after_org = list(
        filter(lambda span: span.start_pos > org_end, pos_tags))
    # If no frame tags are usable, fall back to POS tags.
    if not frame_tags_after_org and not pos_tags_after_org:
        return None

    first_after_org = (
        frame_tags_after_org[0] if frame_tags_after_org else pos_tags_after_org[0]
    )
    original = sentence.to_original_text()
    # Extract reason following ORG occurence.
    reason = original[first_after_org.start_pos:]
    return reason


def clean_organization(full_text: str):
    """ Clean an organisation name (e.g. 'Microsoft Inc.' -> 'Microsoft'). """
    cleaned = full_text.strip().lower() \
        .replace("--", "").replace('"', "").replace("'s", "") \
        .replace("'", "").replace("(", "").replace(")", "")
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
    """ Add a possible reason to the organisation dictionary. If no reason is present,
        count up the organisation appearance count anyways. """
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
    """ Find the top occuring organisations and the reason for their appearance(s). """
    c_top_five = dict(
        sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    )
    r_top_five = dict((item[0], reasons[item[0]])
                      for item in c_top_five.items())
    return r_top_five, c_top_five


def main():
    try:
        if len(sys.argv) < 2:
            print()
            sys.exit(
                "Please supply a path for text processing (e.g. 'CCAT') as an argument for this script."
            )

        reasons, counts = find_organisations_reasons(sys.argv[1])
        top_five_reasons, _ = find_top_five(counts, reasons)
        pretty_print(top_five_reasons)
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
