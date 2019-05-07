import glob
from flair.models import SequenceTagger
from flair.data import Sentence, Span

def clean_organization(org: str):
    original = org
    org = org.strip().lower()
    org = org.replace("--","")
    org = org.replace("\"","")
    org = org.replace("'s","")
    org = org.replace("'","")
    org = org.replace("(","")
    org = org.replace(")","")

    if "," in org:
        org = org.split(",")[0]
    if "." in org: #TODO Move under the length measurement?
        org = org.split(".")[0] 

    org = org.replace(",","")

    if len(org.split(" ")) > 1:
        split = org.split(" ")
        new_org = split[0]
        for s in split[1:]:
            if len(s) > 4:
                new_org = "{} {}".format(new_org, s)
            else: 
                break
        org = new_org
    
    org = org.capitalize()
    print ("{} : {}".format(original, org))
    return org

def get_flair_taggers():
    frame_tagger = SequenceTagger.load('frame-fast')
    ner_tagger = SequenceTagger.load('ner-fast')
    return ner_tagger, frame_tagger

def get_first_organisation(sentence: Sentence) -> Span:
    org_tags = list(filter(lambda span: "ORG" in span.tag, sentence.get_spans('ner')))
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
        #print(path)
        lines = file.readlines()
        for line in lines:
            sentence = Sentence(line)
            ner_tagger.predict(sentence)
            frame_tagger.predict(sentence)
            organisation = get_first_organisation(sentence)
            if not organisation:
                continue

            name = clean_organization(organisation.text)
            reason = get_reason_for_appearance(organisation, sentence)
            if name in organisations and reason:
                organisations[name].append(reason)
            elif reason:
                organisations[name] = [reason]
            else:
                organisations[name] = [] 
                #print("Der blev ikke fundet nogen reason")

    return organisations


orgs = find_organisations("CCAT")
print(orgs.keys)