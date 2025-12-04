# ClinicalNLP2024

Python code for computing LLMs surprisals and linear machine learning models in detecting aphasia

Authors: Yan Cong, Jiyeon Lee, Arianna N. LaCroix

Paper: 
[Leveraging pre-trained large language models for aphasia detection in English and Chinese speakers](https://aclanthology.org/2024.clinicalnlp-1.20)


# Processing TalkBank Files for NLP
**Authors**: Emily Tumacder and Yan Cong

This repository also contains information to process CHAT files and CSVs downloaded from TalkBank (https://talkbank.org/aphasia/index.html). 

## Processing CHAT Files

CHAT files are cleaned using the methods below, based on markers identified in the CHAT manual.
*https://talkbank.org/manuals/CHAT.pdf*

The level of cleaning for utterances is divided into two groups: Cleaned and Semicleaned.

"Cleaned" represents in its most basic and correct form. This means transcriptor replacements replace
the patient's words, repeated words are removed so only one instance remains, partial words spoken by the patient are made full, and nonwords are removed. Filler words are still kept in both cleaned and semicleaned groups.

"Semicleaned" retain's the patients' way of speaking, including repeated words, partial and nonwords, the word before replacement, paraphasias, etc.

See below for a detailed description of what is kept and removed in both cleaned and semicleaned

### Disfluencies Kept

[x] _Word repetition ([/])_ (Semicleaned)

- dog [/] dog --> dog dog

[x] Phrase repeitition (<>[/]) (Semicleaned)

- <that is a> [/] that is a dog --> that is a that is a dog

[x] Patient word revision ([//])

- In semicleaned, both words spoken are kept. In cleaned, only the replacement is kept as it is assumed to be more correct(CHAT)
- a dog [//] beast --> a dog beast

[x] Phrase revision (<>[//])

- In semicleaned, both phrases spoken are kept. In cleaned, only the replacement is kept as it is assumed to be more correct(CHAT)
- <what did you> [//] how can you see it --> what did you how can you see it

[x] Filler words (&-) (Cleaned, Semicleaned)

- &-um --> um

### Disfluencies Removed (Semicleaned and Cleaned)

[x] Pauses (...)
[x] Phonological fragment &+

- &+sn dog (starts with sn-ake sound but switches to dog) --> dog

[x] Gestures (&=)

### Paraphasia (Semicleaned)

[x] Marked with replacement word, sometimes also with an @u

- no dubs [: dogs] [*] allowed --> no dubs allowed
- the pints@u [: prince] [*] wants to know --> the pints wanats to know
- semantic vs. phonetic paraphasia examples: 

### Missing Material, Removed (Semicleaned and Cleaned)

[x] Untranscribed (www)
[x] Unintelligible segments (xx, xxx)

### Shortenings

[x] Leaving off sounds in words

- In semicleaned, the added sound is removed to stay true to the patient's speech. In cleaned, it is added to the word to create a more accurate word
- (be)cause I said so --> cause I said so

### Miscellaneous (Semicleaned and Cleaned)

[x] No punctuation in final output
[x] Replace underscores and plus signs for spaces
you_know --> you know
you+know --> you know
[x] Markers in brackets []

- [+exc] --> " "
- [+gram] --> " "
  [x] Fully alphanumeric
- +//
