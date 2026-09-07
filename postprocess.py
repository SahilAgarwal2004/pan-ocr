import re

# Mapping for number-to-char (for names)
NUM_TO_CHAR = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '9': 'G', '3': 'E', '4': 'A', '7': 'T'
}

# Mapping for char-to-number (for DOB)
CHAR_TO_NUM = {
    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7', 'A': '4', 'E': '3'
}

def correct_ocr_errors(text, field_type):

    text = text.upper().strip()
    if field_type in ["name", "father_name"]:
        # Numbers to chars
        return ''.join([NUM_TO_CHAR.get(c, c) for c in text])
    elif field_type == "dob":
        # Chars to numbers
        return ''.join([CHAR_TO_NUM.get(c, c) for c in text])
    elif field_type == "pan":
        # PAN format: 5 letters, 4 digits, 1 letter
        corrected = []
        for i, c in enumerate(text):
            if i < 5 or i == 9:
                # Should be a letter
                corrected.append(NUM_TO_CHAR.get(c, c))
            elif 5 <= i < 9:
                # Should be a digit
                corrected.append(CHAR_TO_NUM.get(c, c))
            else:
                corrected.append(c)
        return ''.join(corrected)
    return text

def regex_validate(text, field_type):

    if field_type == "pan":
        return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", text))
    elif field_type == "dob":
        return bool(re.fullmatch(r"[0-9]{2}/\[0-9]{2}/\[0-9]{4}", text))
    elif field_type in ["name", "father_name"]:
        # Name: at least 3 letters, all alphabets and spaces
        return bool(re.fullmatch(r"[A-Z ]{3,}", text))
    return False
