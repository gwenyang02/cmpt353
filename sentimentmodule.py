#Module containing function for sentiment analysis
#potentially we could have features sentimentanalysis1 sentimentanalysis2
from rapidfuzz import fuzz
from nltk.corpus import wordnet

def init_sentiment_analyzers():
    """
    Initializer function for each worker process in the Pool.
    Initializes the SentimentIntensityAnalyzer and the Hugging Face sentiment pipeline.
    """
    global sia, sentiment_pipeline
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from transformers import pipeline

    sia = SentimentIntensityAnalyzer()
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model='distilbert-base-uncased-finetuned-sst-2-english',
        # model="SamLowe/roberta-base-go_emotions"
        device=-1
    )
#function sentiment analysis 1 (original approach)
#notes about this approach: (VADER is Inconcistent and so is Hugging Face sometimes, it's better tho)
def sentiment1(inputs):
    """
    Calculates combined sentiment using VADER and Hugging Face's sentiment-analysis pipeline.
    Also checks for mentions of specified politicians.
    """
    # unpack tuple (text, subreddit)
    text, subreddit = inputs

    global sia, sentiment_pipeline

    # Define lists of politicians
    republicans = [
        'Trump', 'Mike','Pence', 'Donald', 'DeSantis', 'McConnell', 'Cruz', 'Rubio',  # Original list
        'Ivanka Trump', 'Donald Trump Jr.', 'Eric Trump', 'Melania Trump',  # Trump family
        'Jared Kushner',  # Advisor
        'Ron DeSantis', 'Mitch McConnell', 'Ted Cruz', 'Marco Rubio',  # Reinforced names
        'Lindsey Graham', 'Kevin McCarthy', 'Marjorie Taylor Greene', 'Lauren Boebert',  # Allies
        'Rudy Giuliani', 'Mike Pompeo', 'Brett Kavanaugh', 'Amy Coney Barrett',  # Advisors/Judges
        'Sean Hannity', 'Tucker Carlson', 'Rush Limbaugh'  # Influential figures
    ]
    democrats = [
        'Hillary', 'Clinton', 'Tim', 'kaine', 'Biden', 'Harris', 'Pelosi', 'Sanders', 'Schumer',  # Original list
        'Bill Clinton', 'Chelsea Clinton',  # Clinton family
        'Joe Biden', 'Jill Biden', 'Hunter Biden', 'Ashley Biden',  # Biden family
        'Kamala Harris', 'Doug Emhoff',  # Harris family
        'Nancy Pelosi', 'Bernie Sanders', 'Chuck Schumer',  # Reinforced names
        'Alexandria Ocasio-Cortez', 'Ilhan Omar', 'Rashida Tlaib', 'Ayanna Pressley',  # Progressive wing
        'Elizabeth Warren', 'Cory Booker', 'Pete Buttigieg',  # Advisors/Key figures
        'Barack Obama', 'Michelle Obama', 'Eric Holder', 'Merrick Garland',  # Obama administration
        'Rachel Maddow', 'Jon Stewart', 'Stephen Colbert'  # Influential figures
    ]

    # Determine VADER sentiment score
    vader_score = sia.polarity_scores(text)['compound']

    # Determine Hugging Face sentiment label
    hf_result = sentiment_pipeline(text[:512])[0]  # Limit text to 512 tokens
    hf_label = hf_result['label']

    # Combine VADER and Hugging Face results
    if hf_label == "POSITIVE":
        combined_score = abs(vader_score)
    else:  # NEGATIVE
        combined_score = -abs(vader_score)

    # edge case: check if "Harris" specifically is mentioned with "bipod"
    # this is a type of gun
    harris_mentioned = "harris" in text.lower()
    bipod_mentioned = "bipod" in text.lower()
    # Check for mentions of Republicans or Democrats
    politician_mentioned = False
    mentioned_group = None

    # Check if "Harris" specifically is mentioned without being around "bipod"
    # and if not around "bipod" and in gun-related subreddits
    # assume they are talking about the gun named harris
    if harris_mentioned and bipod_mentioned:
        if subreddit in ['guns', 'guncontrol', 'Firearms']: # Gun-related subreddits
            poltitician_mentioned = False  # Not Kamala-related
            return combined_score, poltitician_mentioned
        else:
            # Assume Kamala Harris for other subreddits
            mentioned_group = 'democrat'
            combined_score *= -1
            politician_mentioned = True

    if not politician_mentioned:
        for republican in republicans:
            if republican.lower() in text.lower():
                mentioned_group = 'republican'
                combined_score *= 1  # No change for Republicans
                politician_mentioned = True
                break

    if mentioned_group is None:  # Only check Democrats if no Republican match
        for democrat in democrats:
            if democrat.lower() in text.lower():
                mentioned_group = 'democrat'
                combined_score *= -1  # Multiply by -1 for Democrats
                politician_mentioned = True
                break

    # Return both combined_score and politician_mentioned
    return combined_score, politician_mentioned

# -----------
#supporting functions for getting synonyms of the policies
def get_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return synonyms

def precompute_policy_synonyms(policy_list):
    """
    Precompute synonyms for all policies in the policy list.
    Returns a dictionary mapping each policy to its set of synonyms.
    """
    synonym_map = {}
    for policy in policy_list:
        synonyms = get_synonyms(policy)
        synonym_map[policy] = synonyms
    return synonym_map

def find_similar_policy(text, synonym_map, policy_list, fuzzy_threshold=80):
    """
    Find the most similar policy to the input text based on precomputed synonyms and fuzzy matching.
    If no similar policy is found, return 0.
    """
    text_lower = text.lower()

    # First, check for exact or synonym matches
    for policy, synonyms in synonym_map.items():
        all_terms = {policy} | synonyms  # Include both policy and its synonyms
        if any(term.lower() in text_lower for term in all_terms):
            return policy

    # If no synonym match, use fuzzy matching
    for policy in policy_list:
        if fuzz.partial_ratio(policy.lower(), text_lower) >= fuzzy_threshold:
            return policy

    return 0

def sentiment1(text):
    '''
    Calculates normal sentiment score for a post combining VADER and HuggingFace
    :param text: a string
    :return: a float between -1 and 1 representing combined sentiment score for each text
    '''
    global sia, sentiment_pipeline
    # Determine VADER sentiment score
    vader_score = sia.polarity_scores(text)['compound']

    # Determine Hugging Face sentiment label
    hf_result = sentiment_pipeline(text[:512])[0]  # Limit text to 512 tokens
    hf_label = hf_result['label']

    # Combine VADER and Hugging Face results
    if hf_label == "POSITIVE":
        combined_score = abs(vader_score)
    else:  # NEGATIVE
        combined_score = -abs(vader_score)

    return combined_score

# function sentiment analysis 2
def sentiment2(inputs):
    """
    Calculates combined sentiment using VADER and Hugging Face's sentiment-analysis pipeline.
    Also checks for mentions of specified politicians.
    Adds shifting to VADER and Hugging Faccombined sentiment score using a mapping
    for policies related to democrats or republicans
    """

    #importing for string policy variation
    from fuzzywuzzy import fuzz
    from nltk.corpus import wordnet
    from nltk import download
    # Ensure NLTK WordNet is downloaded
    download('wordnet')
    #SHIFTING LOGIC AFTER THE VADER AND HF COMBINED SCORE IS CALCULATED
    #have arrays of positive democrat and positive republican policies
    # so pro democrat = [...Pro choice..] pro republican  = [..freedom of arms..]
    # we use hashtable to see which policy is mentioned
    # hugging face then returns -1 or +1 negative or positive on the sentiment
    # given a negative hugging face on a pro democrat policy, we infer "not a fan of democrats"
    # therefore we add +0.3 some value depending on how extreme the policy is potentially
    # same logic applies for republicans -0.3 if hugging face returns negative on pro republican policy
    # +0.3 if hugging face positive on pro republican
    # - 0.3 if positive on pro democrat comment
    # unpack tuple (text, subreddit)

    text, subreddit = inputs    

    global sia, sentiment_pipeline

    # Define lists of politicians
    republicans = ['Trump', 'Pence', 'DeSantis', 'McConnell', 'Cruz', 'Rubio']
    democrats = ['Hillary', 'Clinton', 'Biden', 'Harris', 'Pelosi', 'Sanders', 'Schumer']

    #policy hashtable
    # expand hashtable to include phrasing related that one party uses
    # e.g. republicans use "politically correct"
    policy_shift = {
        # any positive democrat policy should get a negative!!!!!
        # any positive Republican policy should get a positive!!!
        # Democrat-leaning policies (negative shift for positive sentiment, positive shift for negative sentiment)
        "universal healthcare": -0.5,
        "medicare for all": -0.5,
        "pro-choice": -0.5,
        "climate change action": -0.5,
        "clean energy subsidies": -0.5,
        "solar energy": -0.5,
        "wind power": -0.5,
        "minimum wage increase": -0.5,
        "lgbtq rights": -0.5,
        "same-sex marriage": -0.5,
        "fossil fuel regulation": -0.5,
        "student loan forgiveness": -0.5,
        "tuition-free college": -0.5,
        "wall street reform": -0.5,
        "wealth tax": -0.5,
        "progressive tax": -0.5,
        "public education funding": -0.5,
        "social security expansion": -0.5,
        "universal background checks": -0.5,
        "gun safety": -0.5,
        "black lives matter": -0.5,
        "criminal justice reform": -0.5,
        "marijuana legalization": -0.5,
        "affordable housing": -0.5,
        "paid family leave": -0.5,
        "childcare subsidies": -0.5,
        "dodd-frank act": -0.5,
        "net neutrality": -0.5,
        "paris climate agreement": -0.5,
        "foreign aid": -0.5,
        "diplomacy with iran": -0.5,
        "immigration reform": -0.5,
        "pathway to citizenship": -0.5,
        "obamacare": -0.5,
        "hiring women and minorities": -0.5,
        "public sector unions": -0.5,
        "equal pay": -0.5,
        "expanding food stamps": -0.5,
        "public housing": -0.5,
        "sanctuary cities": -0.5,
        "renewable energy jobs": -0.5,
        "financial regulation": -0.5,
        "campaign finance reform": -0.5,
        "affordable care act": -0.5,
        "closing private prisons": -0.5,
        "carbon tax": -0.5,
        "electric vehicle subsidies": -0.5,
        "reparations for slavery": -0.5,
        "community policing": -0.5,
        "free universal pre-k": -0.5,
        "national paid sick leave": -0.5,
        "worker's rights protections": -0.5,
        "inclusive voting laws": -0.5,
        "expanding medicaid": -0.5,
        "federal arts funding": -0.5,
        "renewable energy tax credits": -0.5,
        "anti-gerrymandering laws": -0.5,
        "affordable insulin initiatives": -0.5,
        "ending fossil fuel subsidies": -0.5,
        "lgbtq+ military service": -0.5,
        "comprehensive sex education": -0.5,
        "ban on assault weapons": -0.5,
        "mandatory police body cameras": -0.5,
        "capping prescription drug prices": -0.5,
        "student debt cancellation": -0.5,
        "environmental justice": -0.5,
        "rural broadband expansion": -0.5,
        "universal basic income": -0.5,
        "child tax credit expansion": -0.5,

        # can manually adjust more
        # Republican-leaning policies (positive shift for positive sentiment, negative shift for negative sentiment)
        "gun rights": +0.5,
        "freedom of arms": +0.5,
        "border security": +0.5,
        "the wall": +0.5,
        "immigration ban": +0.5,
        "defense spending": +0.5,
        "religious freedom": +0.5,
        "school choice": +0.5,
        "tax cuts": +0.5,
        "flat tax": +0.5,
        "capital gains tax cuts": +0.5,
        "privatized healthcare": +0.5,
        "right to work laws": +0.5,
        "repealing obamacare": +0.5,
        "coal mining": +0.5,
        "oil drilling": +0.5,
        "energy independence": +0.5,
        "lowering corporate taxes": +0.5,
        "pro-life": +0.5,
        "opposing abortion": +0.5,
        "traditional marriage": +0.5,
        "military funding": +0.5,
        "voter id laws": +0.5,
        "cutting foreign aid": +0.5,
        "travel ban": +0.5,
        "islamic terrorism": +0.5,
        "american jobs": +0.5,
        "trade tariffs": +0.5,
        "repealing dodd-frank": +0.5,
        "second amendment": +0.5,
        "privatizing social security": +0.5,
        "tough on crime": +0.5,
        "death penalty": +0.5,
        "police funding": +0.5,
        "border patrol": +0.5,
        "anti-refugee policy": +0.5,
        "ending sanctuary cities": +0.5,
        "supporting israel": +0.5,
        "american exceptionalism": +0.5,
        "anti-union laws": +0.5,
        "free market solutions": +0.5,
        "tort reform": +0.5,
        "rebuilding america's military": +0.5,
        "national anthem protests": +0.5,
        "america first": +0.5,
        "drill baby drill": +0.5,
        "refugee restrictions": +0.5,
        "resisting iran nuclear deal": +0.5,
        "against climate regulation": +0.5,
        "abolishing the epa": +0.5,
        "fossil fuel subsidies": +0.5,
        "lowering environmental standards": +0.5,
        "banning transgender bathrooms": +0.5,
        "parental rights in education": +0.5,
        "ban on critical race theory": +0.5,
        "term limits for congress": +0.5,
        "supporting fracking": +0.5,
        "hard borders": +0.5,
        "defunding the irs": +0.5,
        "lowering capital gains tax": +0.5,
        "school vouchers": +0.5,
        "supporting nuclear energy": +0.5,
        "patriotic education": +0.5,
        "constitutional carry": +0.5,
        "pro-business policies": +0.5,
        "ending inheritance tax": +0.5,
        "strengthening border wall": +0.5,
        "law and order": +0.5,
        "defending the second amendment": +0.5,
        "decreasing welfare dependence": +0.5,
        "english as the national language": +0.5,
        "ending affirmative action": +0.5,
        "expanding charter schools": +0.5,
        "protecting traditional values": +0.5,
        "supporting private prisons": +0.5,
        "ban on transgender athletes": +0.5,
        "energy deregulation": +0.5,
        "coal subsidies": +0.5,
        "national voter id laws": +0.5,
        "opposing universal healthcare": +0.5,
        "american energy independence": +0.5,
        "tougher immigration laws": +0.5

    }

    # Case 4: post has no politican and no policy
    # filtering early to reduce unecessary HF computation

    #precomputing policy synonyms
    synonym_map = precompute_policy_synonyms(policy_shift.keys())

    # early filtering for no politician or policy
    contains_party_keywords = any(p.lower() in text.lower() for p in republicans + democrats)
    # check if any policy from policy_shift has a fuzzy match score greater than 80
    #checking if synonym match
    matched_policy = find_similar_policy(text, synonym_map, policy_shift.keys())
    if not contains_party_keywords and matched_policy == 0:
        return 0, False, 0


    # Determine VADER sentiment score
    vader_score = sia.polarity_scores(text)['compound']

    # Determine Hugging Face sentiment label
    hf_result = sentiment_pipeline(text[:512])[0]  # Limit text to 512 tokens
    hf_label = hf_result['label']

    # Combine VADER and Hugging Face results
    if hf_label == "POSITIVE":
        combined_score = abs(vader_score)
    else:  # NEGATIVE
        combined_score = -abs(vader_score)

    # edge case: check if "Harris" specifically is mentioned with "bipod"
    # this is a type of gun
    harris_mentioned = "harris" in text.lower()
    bipod_mentioned = "bipod" in text.lower()
    # Check for mentions of Republicans or Democrats
    politician_mentioned = False
    mentioned_group = None

    # Check for mentions of politicians
    politician_mentioned = False
    for politician in republicans + democrats:
        if politician.lower() in text.lower():
            politician_mentioned = True
            break

    # Check for policy mentions and compute shift
    # -------------------------------
    # case 1: post has politican mentioned and policy
    # what shld happen: caluclate combined sentiment then
    # adjust that by policy fuzzy match shift
    # return the combination (shifted value, i.e. sum)of combined sentiment and shift value
    # case 2: post has politican mentioned and no policy
    # what shld happen: calculate combined sentiment and
    # return that no shifting occurs
    # case 3: post has no politican and has policy
    # skip the polititian mentioned part, just go straight to policy fuzzy match shift,
    # use shift val (has hfint and shift value from map)
    # case 4: post has no politican and no policy
    # filter these out (DO THIS EARLY TO REDUCE HF COMPUTATION)
    shift_val = 0
    for policy, shift in policy_shift.items():
        if fuzz.partial_ratio(policy, text.lower()) > 80:
            #multiply by hugging face value
            if hf_label == "POSITIVE":
                hfint = 1
            else: hfint = -1
            shift_val = shift * hfint
            break

    if politician_mentioned and shift_val != 0:  # Case 1
        shifted = combined_score + shift_val
    elif politician_mentioned and shift_val == 0:  # Case 2
        shifted = combined_score
    elif not politician_mentioned and shift_val != 0:  # Case 3
        combined_score = 0
        shifted = shift_val
    else:  #case4
        return 0, False, 0

    # limit the maximum combined score so that it doesn't go out of bounds from -1 to 1
    # e.g. if a post has sentiment of 0.8, shifting it further by +0.5 would make it go out of bounds
    # do we need to do more standardization of the combined_score after this?
    # combined_score = max(-1, min(1, combined_score))
    shifted = max(-1, min(1, shifted))

    # Return both combined_score, politician_mentioned, shifted
    return combined_score, politician_mentioned, shifted

    # combined score is same as sentiment 1
    # shifted is updated


