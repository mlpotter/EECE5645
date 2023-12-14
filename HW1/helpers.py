import re

def to_lower_case(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'. 
    """
    return s.lower()


def create_list_from_file(filename):
    """Read words from a file and convert them to a list.
   
    Input:
    - filename: The name of a file containing one word per line.
    
    Returns
    - wordlist: a list containing all the words in the file, as strings.

    """
    wordlist = []
    with open(filename) as f:
        line = f.readline()
        while line:
            wordlist.append(line.strip())
            line = f.readline()
        return wordlist     


def strip_non_alpha(s):
    """ Remove non-alphabetic characters from the beginning and end of a string. 

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle 
    of the string should not be removed. E.g. "haven't" should remain unaltered."""
    return re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$","",s)

def is_inflection_of(s1,s2):
    """ Tests if s1 is a common inflection of s2. 

    The function first (a) converts both strings to lowercase and (b) strips
    non-alphabetic characters from the beginning and end of each string. 
    Then, it returns True if the two resulting two strings are equal, or 
    the first string can be produced from the second by adding the following
    endings:
    (a) 's
    (b) s
    (c) es
    (d) ing
    (e) ed
    (f) d
    """
    s1 = strip_non_alpha(s1.lower()); s2 = strip_non_alpha(s2.lower())
    
    endings = ["'s","s","es","ing","ed","d"]
    status = any([s1+end == s2 for end in endings]) or (s1==s2)
    
    return status

def same(s1,s2):
    "Return True if one of the input strings is the inflection of the other."

    return is_inflection_of(s1,s2) | is_inflection_of(s2,s1)

def find_match(word,word_list):
    """Given a word, find a string in a list that is "the same" as this word.

    Input:
    - word: a string
    - word_list: a list of stings

    Return value:
    - A string in word_list that is "the same" as word, None otherwise.
    
    The string word is 'the same' as some string x in word_list, if word is the inflection of x,
    ignoring cases and leading or trailing non-alphabetic characters.
    """
    same_words = [wordi for wordi in word_list if same(word,wordi)]
    if len(same_words) == 0:
        return None
    else:
        return same_words

if __name__=="__main__":
    
    # Test strip_non_alpha
    test_cases = (("!_2bob_@","bob"),
                  ("1haven't","haven't"),
                  ("nowhere@gmail.com\..","nowhere@gmail.com"),
                  ("",""),
                  ("12321.@3",""),
                  ("1a","a"),
                  ("a", "a")
                  )
    for input,output in test_cases:
        # print(input,output,strip_non_alpha(input))
        assert strip_non_alpha(input) == output, (input,output)


    # Test is_inflection_of and same
    test_cases = ((("barts2","barts"),True),
                  (("mope","moped"),True),
                  (("Mill", "Mill's"), True),
                  (("raves", "rave"), False)
                  )
    for input,output in test_cases:
        # print(input,output,is_inflection_of(input[0],input[1]))
        assert is_inflection_of(input[0],input[1]) is output, (input,output)

    test_cases = ((("barts","barts2"),True),
                  (("moped","moped"),True),
                  (("Mill's", "Mill"), True),
                  (("raves", "rave"), True),
                  (("rav312", "rave"), False),
                  (("3_lord", "rave"), False)
                  )
    for input,output in test_cases:
        # print(input,output,is_inflection_of(input[0],input[1]))
        assert same(input[0],input[1]) is output, (input,output)

    # Test find_match
    test_cases = ((("bart",["barts","barted","farted"]),["barts","barted"]),
                  (("barter", ["barts", "barted", "farted"]),None)
                  )
    for input,output in test_cases:
        # print(input,output,is_inflection_of(input[0],input[1]))
        assert find_match(input[0],input[1]) == output, (input,output)