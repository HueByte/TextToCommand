`TfidfVectorizer` for matching commands based on description
`NER` pipeline for extracting entities from the command
`BERT`
`SSM` Sentence Similarity Models



1. Extract the command name based on the prompt and command description via S-BERT (Sentence-Transformer).
2. Extract the potential parameters from the command description via NER. Connect Begin and End tokens.
3. Attempt to match the extracted parameters to the command parameters via S-BERT, refine them and if enough similarity is found, apply the extracted parameter to the currently investigated command parameter
4. Connect extracted command name with the extracted parameters