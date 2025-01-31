# -------------------------------------------------------------------------
# Copyright (c) 2021 Jie Lei
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Source: https://github.com/microsoft/xpretrain/tree/main/CLIP-ViP
# This file is original from the CLIP-ViP project.


"""List of stop words."""
# This list of English stop words is taken from the "Glasgow Information
# Retrieval Group". The original list can be found at
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "actually", "after", "afterwards", "again",
    "against", "all", "almost", "alone", "along", "already", "also", "although",
    "always", "am", "among", "amongst", "amoungst", "amount", "an", "and",
    "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere",
    "are", "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom",
    "but", "by", "call", "can", "cannot", "cant", "can't", "co", "con", "could",
    "couldnt", "cry", "de", "describe", "detail", "do", "done", "don't", "down",
    "due", "during", "each", "easy", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "find",
    "fire", "first", "five", "for", "former", "formerly", "forty", "found",
    "four", "from", "further", "give", "had", "has", "hasnt", "have", "he",
    "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers",
    "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie",
    "if", "i'm", "i'll", "i've", "in", "inc", "indeed", "interest", "is", "it",
    "it'll", "its", "it's", "itself", "just", "keep", "last", "latter",
    "latterly", "least", "less", "like", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "much", "must", "my", "myself", "name", "namely", "neither", "never",
    "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor",
    "not", "nothing", "now", "nowhere", "of", "off", "often", "ok", "okay",
    "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise",
    "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "really", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such", "take",
    "ten", "than", "thank", "thanks", "that", "that's", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "third", "this",
    "those", "though", "three", "through", "throughout", "thru", "thus", "to",
    "together", "too", "top", "toward", "towards", "twelve", "twenty", "two",
    "un", "until", "up", "upon", "us", "very", "via", "view", "viewing",
    "viewer", "was", "we", "we'll", "well", "welcome", "were", "what",
    "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas",
    "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
    "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
    "with", "within", "without", "would", "wont", "won't", "yet", "you", "your",
    "yours", "you've", "you'll", "yourself", "yourselves", "youtube", "going",
    "want", "right", "you're", "we're", "know", "gonna", "need", "bit", "look",
    "yeah", "guys", "sure", "let's", "video", "oh", "let", "today", "they're",
    "did", "looks", "different", "great", "different", "say", "um", "probably",
    "kind", "doesn't", "does", "maybe", "hey", "we've", "better", "hope",
    "there's", "try"
])