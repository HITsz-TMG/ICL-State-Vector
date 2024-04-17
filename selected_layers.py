

def get_data_layers(is_fewshot, is_llama):
    if is_llama:
        if is_fewshot:
            return {
                "person-sport": 12,
                "person-instrument": 12,
                "capitalize": 14,
                "person-occupation": 9,
                "product-company": 11,
                "country-capital": 14,
                "present-past": 12,
                "landmark-country": 12,
                "antonym": 10,
                "english-french": 10,
                "singular-plural": 8,
                "ag_news": 11
            }
        else:
            return {
                "person-sport": 14,
                "person-instrument": 14,
                "capitalize":14,
                "person-occupation":12,
                "product-company":13,
                "country-capital": 14,
                "present-past": 13,
                "landmark-country": 14,
                "antonym":13,
                "english-french": 12,
                "singular-plural": 14,
                "ag_news":14
            }
    else:
        if is_fewshot:
            return {
                "person-occupation": 9,
                "person-instrument": 13,
                "singular-plural": 6,
                "person-sport": 10,
                "capitalize": 9,
                "product-company": 6,
                "country-capital": 11,
                "present-past": 10,
                "landmark-country": 13,
                "antonym": 10,
                "english-french": 8,
                "ag_news": 10
            }
        else:
            return {
                "person-occupation":12,
                "person-instrument": 13,
                "singular-plural": 9,
                "person-sport": 10,
                "capitalize":12,
                "product-company": 9,
                "country-capital": 12,
                "present-past": 13,
                "landmark-country": 13,
                "antonym":12,
                "english-french": 10,
                "ag_news": 13
            }

